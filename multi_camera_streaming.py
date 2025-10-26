import logging
import json
import cv2
import os
import threading
import queue
import datetime
import time
import copy
import socket
import numpy as np

from easymocap.dataset import CONFIG
from easymocap.config.mvmp1f import Config
from easymocap.mytools.camera_utils import Undistort
from easymocap.affinity.affinity import ComposedAffinity
from easymocap.mytools import load_parser, parse_parser
from easymocap.dataset.mvmpmf2 import MVMPMF

from ev_sdk.src.ji import init, get_results

from easymocap.assignment.associate_test import simple_associate
from easymocap.assignment.group_test import PeopleGroup

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# 初始化 GStreamer
Gst.init(None)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义一个队列来存储每个摄像头的图像
queues_annots = queue.Queue(maxsize=1)
# queues_annots = queue.Queue(maxsize=5)
# NOTE 为每个摄像头创建单独的帧缓冲区（单缓冲）
camera_frames = [None, None, None, None]  # 每个摄像头的最新帧
camera_frame_locks = [threading.Lock() for _ in range(4)]  # 每个摄像头的帧锁

img_lock = threading.Lock()
time_locks = [threading.Lock() for _ in range(4)]  # 为每个摄像头添加时间锁

clients = []
lock = threading.Lock()

USE_H264 = False
USE_H265 = True
use_gstreamer_streaming = True  # 启用 GStreamer 推流
mm_time = 0
use_url = True
if use_url:
    urls = [
        "rtsp://admin:kntt13579@192.168.1.201:554/Streaming/Channels/101",
        "rtsp://admin:kntt13579@192.168.1.202:554/Streaming/Channels/101",
        "rtsp://admin:kntt13579@192.168.1.203:554/Streaming/Channels/101",
        "rtsp://admin:kntt13579@192.168.1.204:554/Streaming/Channels/101"
    ]
else:
    urls = [
        "/home/drt/software/Easymocap/data/examples/fengcai/0.avi",
        "/home/drt/software/Easymocap/data/examples/fengcai/1.avi",
        "/home/drt/software/Easymocap/data/examples/fengcai/2.avi",
        "/home/drt/software/Easymocap/data/examples/fengcai/3.avi",
    ]


class GStreamerStreamer(threading.Thread):
    """
    使用 GStreamer 进行视频推流的类
    支持 UDP 和 RTSP 推流
    """
    def __init__(self, camera_id, width=1920, height=1080, fps=30, 
                 output_type='udp', udp_host='127.0.0.1', udp_port=5000):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.output_type = output_type
        self.udp_host = udp_host
        self.udp_port = udp_port + camera_id  # 每个摄像头使用不同端口
        self.running = True
        self.daemon = True
        
        # 创建 GStreamer pipeline
        self.pipeline = None
        self.appsrc = None
        self._create_pipeline()
        
    def _create_pipeline(self):
        """创建 GStreamer 推流 pipeline"""
        self.pipeline = Gst.Pipeline.new(f"streamer_pipeline_{self.camera_id}")
        
        # 创建 appsrc
        self.appsrc = Gst.ElementFactory.make("appsrc", f"source_{self.camera_id}")
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        
        # 设置 caps
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1"
        )
        self.appsrc.set_property("caps", caps)
        
        # 创建其他元素
        videoconvert = Gst.ElementFactory.make("videoconvert", f"convert_{self.camera_id}")
        videoscale = Gst.ElementFactory.make("videoscale", f"scale_{self.camera_id}")
        
        # 使用 NVIDIA 硬件编码器（如果可用）
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder_{self.camera_id}")
        if encoder is None:
            # 回退到软件编码器
            encoder = Gst.ElementFactory.make("x264enc", f"encoder_{self.camera_id}")
            encoder.set_property("tune", "zerolatency")
            encoder.set_property("speed-preset", "ultrafast")
            encoder.set_property("bitrate", 2000)
        else:
            encoder.set_property("bitrate", 2000000)
            encoder.set_property("preset-level", 1)  # UltraFast
        
        h264parse = Gst.ElementFactory.make("h264parse", f"parse_{self.camera_id}")
        
        if self.output_type == 'udp':
            # UDP 推流
            rtppay = Gst.ElementFactory.make("rtph264pay", f"pay_{self.camera_id}")
            rtppay.set_property("config-interval", 1)
            rtppay.set_property("pt", 96)
            
            udpsink = Gst.ElementFactory.make("udpsink", f"sink_{self.camera_id}")
            udpsink.set_property("host", self.udp_host)
            udpsink.set_property("port", self.udp_port)
            udpsink.set_property("sync", False)
            
            # 添加所有元素到 pipeline
            for elem in [self.appsrc, videoconvert, videoscale, encoder, h264parse, rtppay, udpsink]:
                self.pipeline.add(elem)
            
            # 链接元素
            self.appsrc.link(videoconvert)
            videoconvert.link(videoscale)
            videoscale.link(encoder)
            encoder.link(h264parse)
            h264parse.link(rtppay)
            rtppay.link(udpsink)
            
            logger.info(f"Camera {self.camera_id}: UDP streaming to {self.udp_host}:{self.udp_port}")
            
        elif self.output_type == 'rtsp':
            # RTSP 推流 (需要 gst-rtsp-server)
            mpegtsmux = Gst.ElementFactory.make("mpegtsmux", f"mux_{self.camera_id}")
            rtppay = Gst.ElementFactory.make("rtph264pay", f"pay_{self.camera_id}")
            
            tcpserversink = Gst.ElementFactory.make("tcpserversink", f"sink_{self.camera_id}")
            tcpserversink.set_property("host", "0.0.0.0")
            tcpserversink.set_property("port", 8554 + self.camera_id)
            tcpserversink.set_property("sync", False)
            
            # 添加所有元素到 pipeline
            for elem in [self.appsrc, videoconvert, videoscale, encoder, h264parse, rtppay, tcpserversink]:
                self.pipeline.add(elem)
            
            # 链接元素
            self.appsrc.link(videoconvert)
            videoconvert.link(videoscale)
            videoscale.link(encoder)
            encoder.link(h264parse)
            h264parse.link(rtppay)
            rtppay.link(tcpserversink)
            
            logger.info(f"Camera {self.camera_id}: TCP streaming on port {8554 + self.camera_id}")
        
    def run(self):
        """启动推流"""
        self.pipeline.set_state(Gst.State.PLAYING)
        logger.info(f"GStreamer streamer {self.camera_id} started")
        
        while self.running:
            # 获取当前摄像头的帧
            with camera_frame_locks[self.camera_id]:
                if camera_frames[self.camera_id] is None:
                    time.sleep(0.001)
                    continue
                frame = camera_frames[self.camera_id].copy()
            
            # 推送帧到 appsrc
            try:
                # 转换为 GStreamer buffer
                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                
                # 设置时间戳
                buf.pts = self.pipeline.get_clock().get_time() - self.pipeline.get_base_time()
                buf.duration = Gst.SECOND // self.fps
                
                # 推送 buffer
                ret = self.appsrc.emit("push-buffer", buf)
                if ret != Gst.FlowReturn.OK:
                    logger.warning(f"Camera {self.camera_id}: Failed to push buffer")
                    
            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Error pushing frame: {e}")
            
            # 控制帧率
            time.sleep(1.0 / self.fps)
    
    def stop(self):
        """停止推流"""
        self.running = False
        if self.appsrc:
            self.appsrc.emit("end-of-stream")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        logger.info(f"GStreamer streamer {self.camera_id} stopped")


class VideoStream(threading.Thread):

    def __init__(self, video_path, stream_id, flag, width=1920, height=1080):
        super().__init__()
        
        
        if use_url:
            self.stream_id = stream_id
            self.video_path = video_path
            self.flag = flag
            self.width = width
            self.height = height
            self.running = True
            self.frame = None
            self.time = 0
            self.count = 0
            self.frame_id = 0


            # 创建 pipeline
            self.pipeline = Gst.Pipeline.new(f"pipeline_{stream_id}")

            # 创建 uridecodebin 元素
            src = Gst.ElementFactory.make("rtspsrc", f"source_{stream_id}")
            src.set_property("location", video_path)

            # 设置更小的延迟缓冲，避免拉流延迟过大
            src.set_property("latency", 50)  # 设置为50ms
            src.set_property("drop-on-latency", True)  # 丢弃过多的延迟帧

            # 创建解码器、转换器、队列和 appsink
            decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder_{stream_id}")
            converter = Gst.ElementFactory.make("autovideoconvert", f"converter_{stream_id}")
            queue = Gst.ElementFactory.make("queue", f"queue_{stream_id}")
            sink = Gst.ElementFactory.make("appsink", f"sink_{stream_id}")
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)  # 禁用同步来减少延迟
            sink.set_property("max-buffers", 1)  # 限制缓存帧数

            # 设置 caps
            caps = Gst.Caps.from_string("video/x-raw, format=BGR")
            sink.set_property("caps", caps)
            sink.connect("new-sample", self.on_new_sample)

            # 添加所有元素到管道
            self.pipeline.add(src)
            self.pipeline.add(decoder)
            self.pipeline.add(queue)
            self.pipeline.add(converter)
            self.pipeline.add(sink)

            # 连接元素
            src.connect("pad-added", self.cb_newpad, decoder)
            decoder.link(queue)
            queue.link(converter)
            converter.link(sink)
           
            '''
            # 总线监听
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            '''
        else:
            print("==========================use video")
            self.stream_id = stream_id
            self.video_path = video_path  # 使用 URL
            self.flag = flag
            self.width = width
            self.height = height
            self.running = True
            self.frame = None
            self.time = 0
            self.count = 0
            self.frame_id = 0
            self.name = None

            # 创建 GStreamer 管道
            self.pipeline = Gst.Pipeline.new(f"pipeline_{stream_id}")

            # 创建元素

            src = Gst.ElementFactory.make("filesrc", f"source_{stream_id}")  
            src.set_property("location", video_path)
            
            h264parse = Gst.ElementFactory.make("h264parse", f"parser_{stream_id}")
            decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder_{stream_id}")
            converter = Gst.ElementFactory.make("autovideoconvert", f"converter_{stream_id}")#nvvideoconvert autovideoconvert
            sink = Gst.ElementFactory.make("appsink", f"sink_{stream_id}")

            # 设置 appsink 属性
            sink.set_property("emit-signals", True)
            caps = Gst.Caps.from_string(f"video/x-raw, format=BGR, width={width}, height={height}")
            sink.set_property("caps", caps)
            sink.connect("new-sample", self.on_new_sample)

            # 构建管道
            for elem in [src, h264parse, decoder, converter, sink]:
                if elem is None:
                    continue
                self.pipeline.add(elem)
            src.link(h264parse)
            h264parse.link(decoder)
            decoder.link(converter)
            converter.link(sink)
            
            '''
            # 设置总线监听
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            '''
        
        
    def cb_newpad(self, decodebin, pad, decoder):
        caps = pad.get_current_caps()
        gstname = caps.get_structure(0).get_name()
        print(f"Pad added with format: {gstname}")

        if "application/x-rtp" in gstname:
            pipeline = decoder.get_parent()  # 获取pipeline引用
            if USE_H264:
                depayloader = Gst.ElementFactory.make("rtph264depay", "depayloader")
            elif USE_H265:
                depayloader = Gst.ElementFactory.make("rtph265depay", "depayloader")
            
            # 关键修复：将depayloader添加到pipeline
            pipeline.add(depayloader)
            depayloader.sync_state_with_parent()
            
            pad.link(depayloader.get_static_pad("sink"))
            depayloader.link(decoder)
            print(f"RTP stream detected, linking depayloader and decoder.")
        elif "video/x-raw" in gstname:
            pad.link(decoder.get_static_pad("sink"))      
        
    def on_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[Stream {self.stream_id}] Error: {err.message}")
            self.running = False
            self.pipeline.set_state(Gst.State.NULL)
        elif message.type == Gst.MessageType.EOS:
            print(f"[Stream {self.stream_id}] End of Stream")
            self.running = False
            self.pipeline.set_state(Gst.State.NULL)

    def on_new_sample(self, sink):
        
        if self.count  >= 30:
            return Gst.FlowReturn.ERROR
        if self.running==False:
            self.count  += 1
            return Gst.FlowReturn.OK
        
        sample = sink.emit("pull-sample")
        if not sample:
            self.count  += 1
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            self.count  += 1
            return Gst.FlowReturn.OK
        
        # 获取视频帧
        frame = np.ndarray(
            (self.height, self.width, 3), dtype=np.uint8, buffer=map_info.data
        )
        
        # 更新全局帧
        camera_index = {"one":0, "two":1, "three":2, "four":3}.get(self.flag, -1)
        if camera_index >= 0:
            with camera_frame_locks[camera_index]:
                camera_frames[camera_index] = frame.copy()
                now = datetime.datetime.now()
                self.time = now.timestamp()
                self.count = 0
                self.frame = frame
                self.frame_id += 1
                self.name = f"{self.flag}_{now.strftime('%Y%m%d%H%M%S')}{now.microsecond // 1000:03d}_{self.frame_id:07d}.jpg"
                buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    
    

    def run(self):
        
        global camera_frames
        self.pipeline.set_state(Gst.State.PLAYING)
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
       

    def get_frame(self):
        return self.frame

    def get_frame_infos(self):
        return (self.frame, self.time, self.stream_id, self.height, self.width, self.name)

    def stop(self):
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
   


        
        



def write_json(results, frame_id, time, filename, height, width):

    def process_keypoints(raw_kpts):
        """处理关键点数据流"""
        processed = []
        for kid, (x, y, conf) in enumerate(raw_kpts):
            # 坐标为(0, 0)时，直接将置信度设为0
            if x == 0 and y == 0:
                processed.append([0, 0, 0])
            else:
                # 其他情况保持不变
                processed.append([x, y, conf])

        return processed

    data = {
        "filename": filename,
        "width": width,
        "height": height,
        "time": time,
        "annots": [],
        "frameId": frame_id,
        "detect_feet": True,
        "isKeyframe": False
    }
    for cam_index, result in enumerate(results, start=1):
        for person_idx in range(len(result['boxes'])):
            conf = result['boxes'][person_idx]['conf']
            if float(conf[0]) < 0.5:
                continue

            bbox = result['boxes'][person_idx]['xyxy'] + conf
            keypoints = result['keypoints'][person_idx]['data']
            processed_kpts = process_keypoints(keypoints)
            id = result['boxes'][person_idx]['id'][0]
            person_data = {
                "personID": id,
                "bbox": bbox,
                # "keypoints": keypoints[0],
                "keypoints": processed_kpts,
                "isKeyframe": False
            }
            data["annots"].append(person_data)
    return data


def write_json_track(results, frame_id, time, filename, height, width):

    def process_keypoints(raw_kpts):
        """处理关键点数据流"""
        processed = []
        for i in range(0, len(raw_kpts), 3):
            x, y, conf = raw_kpts[i:i + 3]
            # 坐标为(0, 0)时，直接将置信度设为0
            if x == 0 and y == 0:
                processed.append([0, 0, 0])
            else:
                # 其他情况保持不变
                processed.append([x, y, conf])
        return processed

    data = {
        "filename": filename,
        "width": width,
        "height": height,
        "time": time,
        "annots": [],
        "frameId": frame_id,
        "detect_feet": True,
        "isKeyframe": False
    }

    for cam_index, result in enumerate(results, start=1):
        for person_idx in range(len(result)):
            conf = result[person_idx][4]

            if float(conf) < 0.5:
                continue

            bbox = result[person_idx][:5]
            keypoints = result[person_idx][5:80]
            processed_kpts = process_keypoints(keypoints)
            id = int(result[person_idx][-1]) - 1
            person_data = {
                "personID": id,
                "bbox": bbox,
                # "keypoints": keypoints[0],
                "keypoints": processed_kpts,
                "isKeyframe": False
            }
            data["annots"].append(person_data)
    return data


def get_cam(filename):
    flag = filename.split('/')[-1].split('_')[0]
    cam_map = {'one': 0, 'two': 1, 'three': 2, 'four': 3}
    return cam_map.get(flag, -1)


def change_json(json_data, json_data2, personID):
    for list in json_data2['annots']:
        annot = {
            "dump2d": [],
            "id":
            00,
            "keypoints3d": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0]]
        }
        dict_data = {
            "bbox": [0, 0, 0, 0, 0],
            "cam":
            0,
            "keypoints2d": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0]]
        }

        dict_data['bbox'] = list['bbox']
        dict_data['cam'] = get_cam(json_data2['filename'])
        dict_data['keypoints2d'] = list['keypoints']

        if list['personID'] in personID:
            for list2 in json_data['annots']:
                if list2['id'] == list['personID']:
                    list2['dump2d'].append(dict_data)
                    break
        else:
            personID.append(list['personID'])
            annot['dump2d'].append(dict_data)
            annot['id'] = list['personID']
            json_data['annots'].append(annot)

    return json_data


def change_json_data(jsons_2d, json_3d):
    json_data = {
        "filename": jsons_2d[0]['filename'],
        "annots": [],
        "height": jsons_2d[0]['height'],
        "width": jsons_2d[0]['width'],
        "time": jsons_2d[0]['time'],
        "frameId": jsons_2d[0]['frameId']
    }
    # 读取json文件并进行修改
    personID = []
    # 读取2d数据
    for list in jsons_2d[0]['annots']:
        annot = {
            "dump2d": [],
            "id":
            0,
            "keypoints3d": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                            [0, 0, 0, 0]]
        }
        dict_data = {
            "bbox": [0, 0, 0, 0, 0],
            "cam":
            0,
            "keypoints2d": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [0, 0, 0]]
        }

        dict_data['bbox'] = list['bbox']
        dict_data['cam'] = get_cam(jsons_2d[0]['filename'])
        dict_data['keypoints2d'] = list['keypoints']
        annot['dump2d'].append(dict_data)
        annot['id'] = list['personID']
        json_data['annots'].append(annot)
        personID.append(list['personID'])

    for i in range(len(jsons_2d) - 1):
        json_data = change_json(json_data, jsons_2d[i + 1], personID)

    for list in json_3d:
        for list2 in json_data['annots']:
            if int(list2['id']) == int(list['id']):
                list2['keypoints3d'] = list['keypoints3d']
                break

    dicts = json_data
    return dicts



def undistort(images, cameras, cams):
    if cameras is not None and len(images) > 0:
        images_ = []
        for nv in range(len(images)):
            mtx = cameras[cams[nv]]['K']
            dist = cameras[cams[nv]]['dist']
            if images[nv] is not None:
                frame = cv2.undistort(images[nv], mtx, dist, None)
            else:
                frame = None
            images_.append(frame)
    else:
        images_ = images
    return images_


def undis_det(lDetections, cameras, cams=['01', '02', '03', '04']):
    for nv in range(len(lDetections)):
        camera = cameras[cams[nv]]
        for det in lDetections[nv]:
            det['bbox'] = Undistort.bbox(det['bbox'],
                                         K=camera['K'],
                                         dist=camera['dist'])
            keypoints = det['keypoints']
            det['keypoints'] = Undistort.points(keypoints=keypoints,
                                                K=camera['K'],
                                                dist=camera['dist'])
    return lDetections


def get_bbox_from_pose(pose_2d, img=None, rate=0.1):
    validIdx = pose_2d[:, -1] > 0
    if validIdx.sum() == 0:
        return [0, 0, 100, 100, 0]
    y_min = int(min(pose_2d[validIdx, 1]))
    y_max = int(max(pose_2d[validIdx, 1]))
    x_min = int(min(pose_2d[validIdx, 0]))
    x_max = int(max(pose_2d[validIdx, 0]))
    dx = (x_max - x_min) * rate
    dy = (y_max - y_min) * rate
    # 后面加上类别这些
    bbox = [x_min - dx, y_min - dy, x_max + dx, y_max + dy, 1]
    if img is not None:
        correct_bbox(img, bbox)
    return bbox


def correct_bbox(img, bbox):
    w = img.shape[0]
    h = img.shape[1]
    if bbox[2] <= 0 or bbox[0] >= h or bbox[1] >= w or bbox[3] <= 0:
        bbox[4] = 0
    return bbox


def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        try:
            data = json.load(f)
        except:
            logger.error('Reading error {}'.format(path))
            data = []
    return data


def getitem(dataset, annot_input):
    annots = []
    for cam in dataset.cams:
        if dataset.filter2d is not None:
            annot_valid = []
            for ann in annot_input:
                if dataset.filter2d(**ann):
                    annot_valid.append(ann)
            annot = annot_valid
            annot = dataset.filter2d.nms(annot)
        annots.append(annot)
    return annots


def change_annot(data_source, mode='body25'):
    if not isinstance(data_source, list):
        data = data_source['annots']
    else:
        data = data_source
    for i in range(len(data)):
        if 'id' not in data[i].keys():
            data[i]['id'] = data[i].pop('personID')
        if 'keypoints2d' in data[i].keys() and 'keypoints' not in data[i].keys(
        ):
            data[i]['keypoints'] = data[i].pop('keypoints2d')
        for key in [
                'bbox', 'keypoints', 'bbox_handl2d', 'handl2d', 'bbox_handr2d',
                'handr2d', 'bbox_face2d', 'face2d'
        ]:
            if key not in data[i].keys():
                continue
            data[i][key] = np.array(data[i][key])
            if key == 'face2d':
                data[i][key] = data[i][key][17:17 + 51, :]
        if 'bbox' in data[i].keys():
            data[i]['bbox'] = data[i]['bbox'][:5]
            if data[i]['bbox'][-1] < 0.001:
                logger.info('{}/{} bbox conf = 0, may be error'.format(i))
                data[i]['bbox'][-1] = 0
        if mode == 'body25':
            data[i]['keypoints'] = data[i].get('keypoints', np.zeros((25, 3)))
        elif mode == 'body15':
            data[i]['keypoints'] = data[i]['keypoints'][:15, :]
        elif mode in ['handl', 'handr']:
            data[i]['keypoints'] = np.array(data[i][mode + '2d']).astype(
                np.float32)
            key = 'bbox_' + mode + '2d'
            if key not in data[i].keys():
                data[i]['bbox'] = np.array(
                    get_bbox_from_pose(data[i]['keypoints'])).astype(
                        np.float32)
            else:
                data[i]['bbox'] = data[i]['bbox_' + mode + '2d'][:5]
        elif mode == 'total':
            data[i]['keypoints'] = np.vstack([
                data[i][key]
                for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']
            ])
        elif mode == 'bodyhand':
            data[i]['keypoints'] = np.vstack(
                [data[i][key] for key in ['keypoints', 'handl2d', 'handr2d']])
        elif mode == 'bodyhandface':
            data[i]['keypoints'] = np.vstack([
                data[i][key]
                for key in ['keypoints', 'handl2d', 'handr2d', 'face2d']
            ])
        conf = data[i]['keypoints'][..., -1]
        conf[conf < 0] = 0
    data.sort(key=lambda x: x['id'])
    return data


def json_turn(datas):
    # get personID
    personID = []
    max_number_cam = 0
    for personid in datas:
        personID.append(personid)
        max_number_cam = max(max_number_cam, len(datas[personid]))
    outdatas = []
    for cam_id in range(max_number_cam):  # bian li cam
        outdata = []
        for person_id in personID:  # bian li person
            boxes = []
            keypoinst = []
            for index in range(len(datas[person_id])):
                if cam_id == datas[person_id][index]['cam_id']:
                    js_data = datas[person_id][index]
                    box = {
                        "xyxy": [
                            js_data['x'], js_data['y'],
                            js_data['x'] + js_data['width'],
                            js_data['y'] + js_data['height']
                        ],
                        "conf": [js_data['confidence']],
                        "id": [int(person_id) - 1]
                    }
                    boxes.append(box)
                    keypoinst.append({"data": js_data['keypoints2d']})
            data = {
                "boxes": boxes,
                "keypoints": keypoinst,
            }

            outdata.append(data)
        outdatas.append(outdata)

    return outdatas


def draw_keypoints_on_images(imgs, results):
    for cam_idx, img in enumerate(imgs):
        if img is None:
            continue

        for person_id, person_data in results.items():
            for data in person_data:
                if data['cam_id'] == cam_idx:
                    # 绘制边界框
                    x, y, w, h, conf = data['x'], data['y'], data['width'], data[
                        'height'], data['confidence']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # 在左上角显示ID
                    cv2.putText(img, str(person_id), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    conf_text = f"{conf:.2f}"
                    text_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = x + w - text_size[0]
                    text_y = y + text_size[1]
                    cv2.putText(
                        img, conf_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    
                    # 绘制关键点
                    for kpt in data['keypoints2d']:
                        x_pt, y_pt, conf = kpt
                        if conf > 0.01:
                            cv2.circle(img, (int(x_pt), int(y_pt)), 5,
                                       (0, 255, 0), -1)

        output_path = os.path.join('result', f'camera_{cam_idx}.jpg')
        logger.info(output_path)
        cv2.imwrite(output_path, img)


def inference4track(frames, predictor, carmera_matrixs, dataset,
                    affinity_model, group, args, cfg):
    process_flag = True
    if frames is None or len(frames) != 4:
        return 0
    width = 0
    height = 0
    imgs = []
    names = []
    for frame in frames:
        if frame is None or frame[0] is None:
            process_flag = False
            continue
        else:
            height = frame[3]
            width = frame[4]
            imgs.append(frame[0])
            names.append(frame[-1])

    if process_flag and len(imgs) == 4:
        name_str = names[0].split('_')
        frame_id = int(name_str[-1].split('.')[0])
        frame_time = name_str[-2]
        jsons = []
        annots = []
        dicts = []
        time1 = time.time()

        results = get_results(predictor, carmera_matrixs, args.cam_ID,
                              imgs)  # imgs: four image frames
        # draw_keypoints_on_images(imgs, results)
        # os._exit(-1)

        time2 = time.time()
        logger.info(f"process cost average = {(time2 - time1)}")
        if len(results) > 0:
            results = json_turn(results)

            for i, result in enumerate(results):
                json_dicts = write_json(result, frame_id, frame_time, names[i],
                                        height, width)

                jsons.append(copy.deepcopy(json_dicts))
                json_data = change_annot(json_dicts, mode='body25')
                annots.append(getitem(dataset, json_data)[0])
            if args.undis:
                annots = undis_det(annots, dataset.cameras, dataset.cams)

            group.clear()
            affinity, dimGroups = affinity_model(annots)
            results = simple_associate(annots,
                                        affinity,
                                        dimGroups,
                                        dataset.Pall,
                                        group,
                                        cfg=cfg.associate)
            jsons_3d = []
            for pid, people in results.items():
                result = {
                    'id': pid,
                    'keypoints3d': people.keypoints3d.tolist()
                }
                jsons_3d.append(result)
                
            # logger.info(jsons_3d)
            dicts = change_json_data(jsons, jsons_3d)
            # logger.info(dicts)
            #print("jsons_3d = ", jsons_3d)
            # exit(0)
            with img_lock:
                while queues_annots.full():
                    queues_annots.get()  # 队列满时丢弃旧数据
                if dicts:
                    queues_annots.put(dicts)

def broadcast_data(data):
    with lock:
        for client in clients:
            try:
                client.send(data)
            except:
                clients.remove(client)
    
def handle_client(conn, addr):
    BUFFER_SIZE = 4096
    with lock:
        clients.append(conn)
    try:
        while True:
            data = conn.recv(BUFFER_SIZE).decode('ascii')
            if not data:
                print("客户端断开连接")
                break
            if data.__contains__('visualizer'):
                logger.info(
                    "Socket Received 'visualizer', start sending json files..."
                )
                time1 = time.time()
                message_number = 0
                while True:
                    try:
                        time.sleep(0.002)  # 等待一段时间后重试
                        with img_lock:
                            if queues_annots.empty():
                                continue
                            message_number += 1
                            # 发送 JSON 数据，使用 ASCII 编码
                            json_bytes = json.dumps(
                                queues_annots.get()).encode('ascii')
                            length_bytes = (str(len(json_bytes)) +
                                            '\n').encode('ascii')
                            broadcast_data(length_bytes + json_bytes)
                            # conn.send(length_bytes + json_bytes)
                        time2 = time.time()
                        # logger.info(
                        #     f"send message cost average = {(time2-  time1)/message_number}"
                        # )
                        formatted_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        print(f"{formatted_date} send message cost average = {(time2-  time1)/message_number}")
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        break
    except Exception as e:
        print(f"错误: {addr} - {str(e)}")
    finally:
        with lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()


def process_socket():
    HOST = '0.0.0.0'
    PORT = 22346
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(2)
    logger.info(f"Server listening on {HOST}:{PORT}...")

    try:
        while True:
            conn, addr = server_socket.accept()
            logger.info(f"Connected by {addr}")
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\n服务器关闭")
    finally:
        server_socket.close()


if __name__ == "__main__":
    parser = load_parser()
    parser.add_argument(
        "--engine",
        type=str,
        default='../ultralytics/test_weight/best0308s32.engine')
    parser.add_argument('--vis_match', action='store_true')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--calibration_path', type=str, default='none')
    parser.add_argument('--cam_ID', type=int, default= 0)
    parser.add_argument('--vis3d', action='store_true')
    parser.add_argument('--ret_crop', action='store_true')
    parser.add_argument("--host", type=str, default='none')  # cn0314000675l
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--stream_type", type=str, default='udp', choices=['udp', 'rtsp'])
    parser.add_argument("--stream_host", type=str, default='127.0.0.1')
    parser.add_argument("--stream_port", type=int, default=5000)
    args = parse_parser(parser)

    cfg = Config.load(args.cfg, args.cfg_opts)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if not os.path.exists(os.path.join(args.out, 'all')):
        os.makedirs(os.path.join(args.out, 'all'))
    if not os.path.exists(args.images):
        os.makedirs(args.images)
    if not os.path.exists(os.path.join(args.images, '01')):
        os.makedirs(os.path.join(args.images, '01'))
    if not os.path.exists(os.path.join(args.images, '02')):
        os.makedirs(os.path.join(args.images, '02'))
    if not os.path.exists(os.path.join(args.images, '03')):
        os.makedirs(os.path.join(args.images, '03'))
    if not os.path.exists(os.path.join(args.images, '04')):
        os.makedirs(os.path.join(args.images, '04'))

    dataset = MVMPMF(args.path,
                     cams=['01', '02', '03', '04'],
                     annot_root=args.annot,
                     config=CONFIG[args.body],
                     kpts_type=args.body,
                     undis=args.undis,
                     no_img=True,
                     out=args.out,
                     filter2d=cfg.dataset)
    dataset.no_img = not (args.vis_det or args.vis_match or args.vis_repro
                          or args.ret_crop)
    affinity_model = ComposedAffinity(cameras=dataset.cameras,
                                      basenames=dataset.cams,
                                      cfg=cfg.affinity)
    group = PeopleGroup(Pall=dataset.Pall, cfg=cfg.group)
    
    # 启动视频拉流线程
    streams = []
    flags = ['one', 'two', 'three', 'four']
    for i, video_path in enumerate(urls):
        stream = VideoStream(video_path, i, flags[i])
        streams.append(stream)
        stream.start()

    # 启动 Socket 服务器线程（用于发送 JSON 数据）
    process_thread2 = threading.Thread(target=process_socket)
    process_thread2.daemon = True
    process_thread2.start()
    
    # 启动 GStreamer 推流线程（替代 FastAPI）
    gst_streamers = []
    if use_gstreamer_streaming:
        logger.info(f"Starting GStreamer {args.stream_type.upper()} streaming...")
        for i in range(4):
            streamer = GStreamerStreamer(
                camera_id=i,
                width=1920,
                height=1080,
                fps=30,
                output_type=args.stream_type,
                udp_host=args.stream_host,
                udp_port=args.stream_port
            )
            gst_streamers.append(streamer)
            streamer.start()
            
        # 打印推流信息
        if args.stream_type == 'udp':
            logger.info("=" * 60)
            logger.info("GStreamer UDP Streaming Started!")
            logger.info("To view streams, use the following GStreamer commands:")
            for i in range(4):
                port = args.stream_port + i
                logger.info(f"Camera {i}: gst-launch-1.0 udpsrc port={port} ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink")
            logger.info("=" * 60)
        elif args.stream_type == 'rtsp':
            logger.info("=" * 60)
            logger.info("GStreamer TCP Streaming Started!")
            for i in range(4):
                port = 8554 + i
                logger.info(f"Camera {i}: Available on TCP port {port}")
            logger.info("=" * 60)
    
    predictor, carmera_matrixs = init(args.calibration_path, args.cam_ID)
    # 启动一个线程来识别图像 2d+3d
    # 主线程等待
    old_name = None
    try:
        while True:
            frames = []
            for stream in streams:
                frame = stream.get_frame_infos()  # six infos
                if frame is not None and frame[0] is not None:
                    frames.append(frame)
            # 如果每个视频流都能获取到帧，显示它们
            if len(frames) == 4:
                if frames[0][-1] != old_name:
                    time1 = time.time()
                    inference4track(frames, predictor, carmera_matrixs, dataset,
                                    affinity_model, group, args, cfg)
                    time2 = time.time()
                    # logger.info(f"yolov8 cost average = {(time2 - time1)}")
                    old_name = frames[0][-1]
                else:
                    time.sleep(0.001)
                    continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Exiting...")
        
        # 停止所有线程
        for stream in streams:
            stream.stop()
            stream.join()
        
        for streamer in gst_streamers:
            streamer.stop()
            streamer.join()

        process_thread2.join()
