import wx
import wx.grid
import sqlite3
import os
import io
import zlib
import datetime
import time
import traceback
import logging
import threading
import numpy as np
import cv2
import pandas as pd
import pyttsx3
import queue
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import dlib
import random


# 设置 Matplotlib 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(filename='attendance.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 菜单项 ID
ID_NEW_REGISTER = 160
ID_FINISH_REGISTER = 161
ID_START_PUNCHCARD = 190
ID_END_PUNCHCARD = 191
ID_OPEN_LOGCAT = 283
ID_CLOSE_LOGCAT = 284
ID_EXPORT_LOGCAT = 285
ID_OPEN_DASHBOARD = 286
ID_MANAGE_USERS = 287
ID_ADD_USER = 288
ID_DELETE_USER = 289
ID_EDIT_USER = 290
ID_SET_WORK_HOURS = 291
ID_MAKEUP_PUNCHCARD = 292

# 员工状态 ID
ID_WORKER_UNAVAILABLE = -1

# 路径
PATH_FACE = "data/face_img_database/"
PATH_PUNCHCARD_PHOTOS = "data/punchcard_photos/"
MODEL_SHAPE_PREDICTOR = "model/shape_predictor_68_face_landmarks.dat"
MODEL_FACE_RECOGNITION = "model/dlib_face_recognition_resnet_model_v1.dat"

if not os.path.exists(PATH_FACE):
    os.makedirs(PATH_FACE)
    logging.info(f"创建目录: {PATH_FACE}")
if not os.path.exists(PATH_PUNCHCARD_PHOTOS):
    os.makedirs(PATH_PUNCHCARD_PHOTOS)
    logging.info(f"创建目录: {PATH_PUNCHCARD_PHOTOS}")

# 活体检测常量
EAR_THRESHOLD = 0.18
EAR_CONSEC_FRAMES = 1
REQUIRED_BLINKS = 2
LIVENESS_STEP_TIMEOUT = 10.0
FRAME_RATE = 30

# 活体检测状态
LIVENESS_STATE_IDLE = 0
LIVENESS_STATE_WAIT_FACE = 1
LIVENESS_STATE_WAIT_BLINK = 2
LIVENESS_STATE_PASSED = 5
LIVENESS_STATE_FAILED = 6
LIVENESS_STATE_PROCESSING = 7

# 辅助函数
def add_mask(image, mask_path="mask.png"):
    try:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            raise FileNotFoundError("无法加载口罩图像")
        
        height, width, _ = image.shape
        mask_y = int(height * 0.5)
        mask_height = int(height * 0.25)
        mask_width = int(width * 0.5)
        mask_x = (width - mask_width) // 2
        
        mask_resized = cv2.resize(mask_img, (mask_width, mask_height), interpolation=cv2.INTER_AREA)
        
        if mask_resized.shape[2] == 4:
            mask_rgb = mask_resized[:, :, :3]
            alpha = mask_resized[:, :, 3] / 255.0
        else:
            mask_rgb = mask_resized
            alpha = np.ones((mask_height, mask_width))
        
        for c in range(3):
            image[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width, c] = (
                image[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width, c] * (1 - alpha) +
                mask_rgb[:, :, c] * alpha
            )
        return image
    except Exception as e:
        logging.warning(f"无法叠加口罩图像: {e}，使用黑色矩形")
        height, width, _ = image.shape
        mask_width = random.randint(int(width * 0.3), int(width * 0.5))
        mask_height = random.randint(int(height * 0.1), int(height * 0.2))
        mask_x = random.randint(0, width - mask_width)
        mask_y = int(height * 0.5)
        image[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width] = (0, 0, 0)
        return image

def return_euclidean_distance(feature_1, feature_2, is_masked=False):
    if not isinstance(feature_1, np.ndarray) or not isinstance(feature_2, np.ndarray):
        return float('inf')
    if feature_1.shape != feature_2.shape:
        return float('inf')
    threshold = 0.3 if is_masked else 0.5
    dist = np.linalg.norm(feature_1 - feature_2)
    logging.info(f"特征距离: {dist:.4f}, 阈值: {threshold}, 是否遮挡: {is_masked}")
    return "same" if dist < threshold else "diff"

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if not os.path.exists(font_path):
            font_path = os.path.join(os.getcwd(), "simhei.ttf")
        if not os.path.exists(font_path):
            raise FileNotFoundError("SimHei 字体未找到")
        font = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    except (IOError, FileNotFoundError):
        font = ImageFont.load_default()
        logging.warning("未找到 SimHei 字体，使用默认字体")
    draw.text(position, text, font=font, fill=textColor)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def check_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 40

# 主应用程序类
class WAS(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, parent=None, title="员工考勤系统", size=(1280, 720))
        self.SetMinSize((800, 600))
        self.SetBackgroundColour('pale blue')
        self.language = 'zh'
        self.translations = {
            'zh': {
                'welcome': "系统启动成功，欢迎使用员工考勤系统！",
                'start_punch': "正在启动刷脸签到...",
                'camera_open': "摄像头已打开，请将脸部正对摄像头（距离 30-50 厘米）。",
                'blink_prompt': "请缓慢眨眼 {} 次，每次闭眼保持 0.2 秒",
                'timeout': "活体检测超时，请重试。",
                'retry': "请重新对准摄像头并缓慢眨眼。",
                'lighting': "光线不足，请调整环境或打开灯光。",
                'face_lost': "人脸丢失，请保持脸部正对摄像头，距离 30-50 厘米。",
                'success': "检测到为真人",
                'register_start': "开始为工号 {}, 姓名 '{}' 采集人脸。",
                'unrecognized': "未识别到已注册用户，请先录入人脸。",
                'repeated_checkin': "请勿重复签到（60 秒内）。",
                'model_error': "人脸识别模型未加载，请检查配置。",
                'blink_progress': "已完成眨眼：{}/{}",
                'error': "错误：{}",
                'login_failed': "用户名或密码错误",
                'punch_success': "签到成功: {} (工号: {})",
                'adjust_camera': "请将脸部对准摄像头进行签到",
                'image_captured': "已捕获图像 {}/10",
                'register_success': "人脸录入成功: 工号 {}, 姓名 {}",
                'register_failed': "人脸录入失败，未检测到有效人脸",
                'multiple_faces_detected': "检测到多张人脸，请确保只有一人",
                'face_not_detected': "未检测到人脸，请调整位置",
                'mask_detected': "检测到可能遮挡，尝试使用口罩模式识别",
                'set_work_hours': "设置上下班时间",
                'work_hours_set': "上下班时间已设置为: 上班 {}, 下班 {}",
                'no_admin_priv': "您没有管理员权限"
            }
        }
        # 初始化语音引擎
        self.speech_engine = pyttsx3.init()
        voices = self.speech_engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.name.lower() or 'chinese' in voice.name.lower():
                self.speech_engine.setProperty('voice', voice.id)
                break
        self.speech_engine.setProperty('rate', 250)
        self.speech_engine.setProperty('volume', 1.0)
        # 初始化语音队列
        self.speech_queue = queue.Queue(maxsize=5)
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        # 初始化 Dlib 模型
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(MODEL_SHAPE_PREDICTOR)
            self.face_recognizer = dlib.face_recognition_model_v1(MODEL_FACE_RECOGNITION)
            logging.info("Dlib 模型加载成功")
        except Exception as e:
            logging.error(f"加载 Dlib 模型失败: {e}")
            self.face_detector = None
            wx.MessageBox(f"无法加载 Dlib 模型: {e}\n请检查模型文件是否存在", "错误", wx.OK | wx.ICON_ERROR)
        # 设置颜色变换定时器
        self.color_list = ['light blue', 'light green', 'light gray', 'cornsilk']
        self.color_index = 0
        self.color_timer = wx.Timer(self, id=wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.color_timer)
        self.color_timer.Start(3000)
        # 绑定窗口关闭事件
        self.Bind(wx.EVT_CLOSE, self.OnExit)
        self.Bind(wx.EVT_SIZE, self.OnResize)
        self.initDatabase()
        self.initUI()
        self.initMenu()
        self.initData()

    def _speech_worker(self):
        while True:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text is None:
                    break
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.warning(f"语音提示失败: {e}")
                if hasattr(self, 'infoText') and self.infoText:
                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + f"语音提示失败: {e}\r\n")
                self.speech_queue.task_done()

    def speak(self, text, priority=False):
        if not hasattr(self, 'speech_queue'):
            return
        if self.speech_queue.full() and not priority:
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                pass
        try:
            self.speech_queue.put(text, timeout=0.1)
            if hasattr(self, 'infoText') and self.infoText:
                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + f"添加语音播报：{text}，队列长度：{self.speech_queue.qsize()}\r\n")
        except queue.Full:
            logging.warning("语音队列已满，消息未添加")
            if hasattr(self, 'infoText') and self.infoText:
                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + "语音队列已满，当前消息未播报\r\n")

    def translate(self, key, default=None):
        return self.translations['zh'].get(key, default or key)

    def getDateAndTime(self):
        return time.strftime("[%Y-%m-%d %H:%M:%S] ")

    def get_user_role(self):
        if not hasattr(self, 'user_role') or self.user_role is None:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM users WHERE username = ?", (self.current_username,))
            result = cursor.fetchone()
            self.user_role = result[0] if result else None
            conn.close()
        return self.user_role

    def initUI(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.status_panel = wx.Panel(self, size=(300, 500))
        self.status_panel.SetBackgroundColour('white')
        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.blink_label = wx.StaticText(self.status_panel, label="眨眼进度: 0/2", pos=(10, 10))
        self.blink_label.SetFont(font)
        self.progress_bar = wx.Gauge(self.status_panel, range=100, pos=(10, 40), size=(280, 30))
        top_sizer.Add(self.status_panel, 0, wx.EXPAND | wx.ALL, 5)

        img_width, img_height = 640, 480
        img = wx.Image(img_width, img_height)
        placeholder_color = (200, 200, 200)
        buffer_data = bytearray([placeholder_color[0], placeholder_color[1], placeholder_color[2]] * (img_width * img_height))
        img.SetData(buffer_data)
        default_bitmap = wx.Bitmap(img)
        self.bmp = wx.StaticBitmap(self, size=(640, 480), bitmap=default_bitmap)
        self.bmp.SetDoubleBuffered(True)
        self.default_bitmap = default_bitmap
        top_sizer.Add(self.bmp, 1, wx.EXPAND | wx.ALL, 5)

        self.sizer.Add(top_sizer, 1, wx.EXPAND | wx.ALL, 5)

        self.infoText = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.infoText.SetFont(font)
        self.infoText.AppendText(self.translate("welcome") + "\n")
        if self.face_detector is None:
            self.infoText.AppendText(self.getDateAndTime() + "人脸识别功能不可用（模型未加载）\r\n")
        else:
            self.infoText.AppendText(self.getDateAndTime() + "人脸识别功能已启用\r\n")
        self.sizer.Add(self.infoText, 0, wx.EXPAND | wx.ALL, 5)

        self.Layout()

    def OnResize(self, event):
        if hasattr(self, 'sizer'):
            self.Layout()
        if hasattr(self, 'dashboard') and self.dashboard and self.dashboard.IsShown():
            self.dashboard.Layout()
        event.Skip()

    def initDatabase(self):
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS worker_info (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    face_feature BLOB,
                    username TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logcat (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id INTEGER,
                    worker_name TEXT,
                    punchcard_datetime TEXT,
                    photo_path TEXT,
                    username TEXT,
                    punch_type TEXT,
                    status TEXT
                )
            ''')
            cursor.execute("PRAGMA table_info(logcat)")
            columns = [col[1] for col in cursor.fetchall()]
            for col in ['photo_path', 'username', 'punch_type', 'status']:
                if col not in columns:
                    cursor.execute(f"ALTER TABLE logcat ADD COLUMN {col} TEXT")
            cursor.execute("PRAGMA table_info(worker_info)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'username' not in columns:
                cursor.execute("ALTER TABLE worker_info ADD COLUMN username TEXT")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password TEXT,
                    role TEXT
                )
            ''')
            cursor.execute("INSERT OR IGNORE INTO users (user_id, username, password, role) VALUES (?, ?, ?, ?)", 
                           (1, 'admin', 'admin123', 'admin'))
            cursor.execute("INSERT OR IGNORE INTO users (user_id, username, password, role) VALUES (?, ?, ?, ?)", 
                           (2, 'user1', 'user123', 'user'))
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS work_hours (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT,
                    end_time TEXT,
                    username TEXT UNIQUE
                )
            ''')
            cursor.execute("INSERT OR IGNORE INTO work_hours (id, start_time, end_time, username) VALUES (?, ?, ?, ?)", 
                           (1, '09:00', '18:00', 'admin'))
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_worker_id ON worker_info(id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logcat_datetime ON logcat(punchcard_datetime)')
            conn.commit()
            conn.close()
            logging.info("数据库初始化成功")
        except Exception as e:
            logging.error(f"数据库初始化失败: {e}")
            traceback.print_exc()

    def initData(self):
        self.name = ""
        self.id = ID_WORKER_UNAVAILABLE
        self.pic_num = 0
        self.flag_registed = False
        self.cap = None
        self.grid_logcat = None
        self.dashboard = None
        self._is_finishing_register = False
        self._is_finishing_punchcard = False
        self.liveness_state = LIVENESS_STATE_IDLE
        self.liveness_start_time = 0.0
        self.blink_consec_frames = 0
        self.blink_count = 0
        self.current_liveness_prompt = ""
        self.knew_id = []
        self.knew_name = []
        self.knew_face_feature = []
        self.face_lost_time = 0.0
        self.current_username = None
        self.work_start_time = None
        self.work_end_time = None
        self.face_box_display_time = 0.0
        self.last_detected_face = None
        self.loadDataBase(1)
        self.loadWorkHours()

    def initMenu(self):
        username = wx.GetTextFromUser("请输入用户名", "登录")
        password = wx.GetPasswordFromUser("请输入密码", "登录")
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (username, password))
        result = cursor.fetchone()
        conn.close()
        if not result:
            wx.MessageBox(self.translate("login_failed"), "错误", wx.OK | wx.ICON_ERROR)
            self.Close()
            return
        self.user_role = result[0]
        self.current_username = username

        role_display = "管理员" if self.user_role == "admin" else "普通用户"
        self.infoText.AppendText(self.getDateAndTime() + f"当前用户：{self.current_username}（{role_display}）\r\n")

        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()

        if self.user_role == "admin":
            self.new_register = fileMenu.Append(ID_NEW_REGISTER, "新建录入")
            self.finish_register = fileMenu.Append(ID_FINISH_REGISTER, "完成录入")
            self.set_work_hours = fileMenu.Append(ID_SET_WORK_HOURS, "设置上下班时间")
            self.makeup_punchcard = fileMenu.Append(ID_MAKEUP_PUNCHCARD, "补卡功能")
            self.Bind(wx.EVT_MENU, self.OnNewRegisterClicked, id=ID_NEW_REGISTER)
            self.Bind(wx.EVT_MENU, self.OnFinishRegisterClicked, id=ID_FINISH_REGISTER)
            self.Bind(wx.EVT_MENU, self.OnSetWorkHoursClicked, id=ID_SET_WORK_HOURS)
            self.Bind(wx.EVT_MENU, self.OnMakeupPunchCard, id=ID_MAKEUP_PUNCHCARD)

        self.start_punchcard = fileMenu.Append(ID_START_PUNCHCARD, "开始签到")
        self.end_punchcard = fileMenu.Append(ID_END_PUNCHCARD, "结束签到")
        fileMenu.AppendSeparator()
        fileMenu.Append(wx.ID_EXIT, "退出")
        menuBar.Append(fileMenu, "文件")

        logcatMenu = wx.Menu()
        logcatMenu.Append(ID_OPEN_LOGCAT, "打开考勤日志")
        logcatMenu.Append(ID_CLOSE_LOGCAT, "关闭考勤日志")
        logcatMenu.Append(ID_EXPORT_LOGCAT, "导出考勤日志")
        #logcatMenu.Append(ID_OPEN_DASHBOARD, "数据仪表板")
        menuBar.Append(logcatMenu, "日志")

        if self.user_role == "admin":
            userMenu = wx.Menu()
            userMenu.Append(ID_MANAGE_USERS, "用户管理")
            menuBar.Append(userMenu, "用户管理")
            self.Bind(wx.EVT_MENU, self.OnManageUsersClicked, id=ID_MANAGE_USERS)

        self.Bind(wx.EVT_MENU, self.OnStartPunchCardClicked, id=ID_START_PUNCHCARD)
        self.Bind(wx.EVT_MENU, self.OnEndPunchCard, id=ID_END_PUNCHCARD)
        self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.OnOpenLogcatClicked, id=ID_OPEN_LOGCAT)
        self.Bind(wx.EVT_MENU, self.OnCloseLogcatClicked, id=ID_CLOSE_LOGCAT)
        self.Bind(wx.EVT_MENU, self.OnExportLogcatClicked, id=ID_EXPORT_LOGCAT)
        self.Bind(wx.EVT_MENU, self.OnOpenDashboardClicked, id=ID_OPEN_DASHBOARD)

        if self.user_role == "admin":
            if self.face_detector is None:
                self.new_register.Enable(False)
                self.finish_register.Enable(False)
                self.set_work_hours.Enable(False)
                self.makeup_punchcard.Enable(False)
                self.infoText.AppendText(self.getDateAndTime() + "管理员模式：人脸录入功能已禁用（模型未加载）\r\n")
            else:
                self.new_register.Enable(True)
                self.finish_register.Enable(False)
                self.set_work_hours.Enable(True)
                self.makeup_punchcard.Enable(True)
                self.infoText.AppendText(self.getDateAndTime() + "管理员模式：人脸录入功能已启用\r\n")

        if self.face_detector is None:
            self.start_punchcard.Enable(False)
            self.end_punchcard.Enable(False)
            self.infoText.AppendText(self.getDateAndTime() + "签到功能已禁用（模型未加载）\r\n")
        else:
            self.start_punchcard.Enable(True)
            self.end_punchcard.Enable(False)
            self.infoText.AppendText(self.getDateAndTime() + "签到功能已启用\r\n")

        self.SetMenuBar(menuBar)
        self.Refresh()
        self.Update()

    def OnTimer(self, event):
        if hasattr(self, 'infoText') and self.infoText and self.infoText.IsShown():
            self.color_index = (self.color_index + 1) % len(self.color_list)
            self.infoText.SetBackgroundColour(self.color_list[self.color_index])
            self.infoText.Refresh()

    def loadDataBase(self, mode):
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            if mode == 1:
                cursor.execute("SELECT id, name, face_feature FROM worker_info WHERE username = ? OR username IS NULL", (self.current_username,))
                self.knew_id = []
                self.knew_name = []
                self.knew_face_feature = []
                for row in cursor.fetchall():
                    self.knew_id.append(row[0])
                    self.knew_name.append(row[1])
                    decompressed_data = zlib.decompress(row[2])
                    feature_array = np.frombuffer(decompressed_data, dtype=np.float64)
                    self.knew_face_feature.append(feature_array)
                logging.info(f"加载了 {len(self.knew_id)} 条员工数据")
            conn.close()
        except Exception as e:
            logging.error(f"加载数据库失败: {e}")
            self.infoText.AppendText(self.getDateAndTime() + self.translate("error").format(f"加载数据库失败 - {e}") + "\r\n")
            traceback.print_exc()

    def loadWorkHours(self):
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT start_time, end_time FROM work_hours WHERE username = ?", (self.current_username,))
            result = cursor.fetchone()
            if result:
                self.work_start_time, self.work_end_time = result
                self.infoText.AppendText(self.getDateAndTime() + f"当前上下班时间: 上班 {self.work_start_time}, 下班 {self.work_end_time}\r\n")
            else:
                self.work_start_time = '09:00'
                self.work_end_time = '18:00'
                self.infoText.AppendText(self.getDateAndTime() + f"使用默认上下班时间: 上班 {self.work_start_time}, 下班 {self.work_end_time}\r\n")
            conn.close()
        except Exception as e:
            logging.error(f"加载上下班时间失败: {e}")
            self.infoText.AppendText(self.getDateAndTime() + self.translate("error").format(f"加载上下班时间失败 - {e}") + "\r\n")
            self.work_start_time = '09:00'
            self.work_end_time = '18:00'

    def insertARow(self, data, mode):
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            if mode == 1:
                id, name, feature = data
                compressed_data = zlib.compress(feature.tobytes())
                cursor.execute("INSERT INTO worker_info (id, name, face_feature, username) VALUES (?, ?, ?, ?)", 
                               (id, name, compressed_data, self.current_username))
                logging.info(f"插入员工数据: 工号={id}, 姓名={name}")
            elif mode == 2:
                worker_id, worker_name, punchcard_datetime, photo_path, punch_type, status = data
                cursor.execute("INSERT INTO logcat (worker_id, worker_name, punchcard_datetime, photo_path, username, punch_type, status) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                               (worker_id, worker_name, punchcard_datetime, photo_path, self.current_username, punch_type, status))
                logging.info(f"插入签到记录: 工号={worker_id}, 姓名={worker_name}, 时间={punchcard_datetime}, 类型={punch_type}, 状态={status}")
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"插入数据库记录失败: {e}")
            self.infoText.AppendText(self.getDateAndTime() + self.translate("error").format(f"插入数据库记录失败 - {e}") + "\r\n")
            traceback.print_exc()

    def get_eye_points(self, shape):
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
        return left_eye, right_eye
    
    def register_cap(self, event):
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("error").format("无法打开摄像头") + "\r\n")
            wx.CallAfter(self.OnFinishRegisterClicked, event)
            return

        features = []
        last_brightness_warning = 0
        last_bitmap_update = 0
        update_interval = 0.1

        while self.cap.isOpened() and not self._is_finishing_register and self.pic_num < 10:
            try:
                frame_start = time.time()
                current_time = time.time()
                flag, im_rd = self.cap.read()
                if not flag or im_rd is None:
                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("警告：摄像头读取失败，尝试重新读取...") + "\r\n")
                    time.sleep(0.1)
                    continue
                im_rd = cv2.flip(im_rd, 1)
                img_rgb = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)

                if current_time - last_brightness_warning > 3:
                    if check_brightness(im_rd):
                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("lighting") + "\r\n")
                        self.speak(self.translate("lighting"))
                        last_brightness_warning = current_time
                        time.sleep(1)
                        continue

                img_masked = add_mask(img_rgb.copy()) if self.pic_num % 2 == 0 else img_rgb

                gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)
                faces = self.face_detector(gray, 1)
                if len(faces) == 1:
                    face = faces[0]
                    shape = self.shape_predictor(img_masked, face)
                    feature = np.array(self.face_recognizer.compute_face_descriptor(img_masked, shape))
                    features.append(feature)
                    self.pic_num += 1
                    cv2.rectangle(im_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("image_captured").format(self.pic_num) + "\r\n")
                    self.speak(self.translate("image_captured").format(self.pic_num))
                    img_path = os.path.join(PATH_FACE, self.name, f"{self.pic_num}.jpg")
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(img_path, cv2.cvtColor(img_masked, cv2.COLOR_RGB2BGR))
                elif len(faces) > 1:
                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("multiple_faces_detected") + "\r\n")
                    self.speak(self.translate("multiple_faces_detected"))
                else:
                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("face_not_detected") + "\r\n")
                    self.speak(self.translate("face_not_detected"))

                if current_time - last_bitmap_update >= update_interval:
                    buf = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB).tobytes()
                    w, h = im_rd.shape[1], im_rd.shape[0]
                    img = wx.Image(w, h, buf)
                    wx_bitmap = wx.Bitmap(img)
                    if hasattr(self, 'bmp') and self.bmp and self.bmp.IsShown():
                        wx.CallAfter(self.bmp.SetBitmap, wx_bitmap)
                    last_bitmap_update = current_time

                frame_time = time.time() - frame_start
                sleep_time = max(0, (1.0 / FRAME_RATE) - frame_time)
                time.sleep(sleep_time)
            except Exception as e:
                error_msg = self.translate("error").format(f"人脸录入异常 - {e}")
                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + error_msg + "\r\n")
                logging.error(f"人脸录入异常: {e}")
                traceback.print_exc()
                break

        if features:
            avg_feature = np.mean(features, axis=0)
            self.insertARow([self.id, self.name, avg_feature], 1)
            self.flag_registed = True
            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("register_success").format(self.id, self.name) + "\r\n")
            wx.CallAfter(wx.MessageBox, self.translate("register_success").format(self.id, self.name), "录入成功", wx.OK | wx.ICON_INFORMATION)
            self.speak(self.translate("register_success").format(self.id, self.name), priority=True)
        else:
            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("register_failed") + "\r\n")
            self.speak(self.translate("register_failed"), priority=True)

        if not self._is_finishing_register:
            wx.CallAfter(self.OnFinishRegisterClicked, event)
    

    def OnNewRegisterClicked(self, event):
        self.infoText.AppendText(self.getDateAndTime() + "点击了新建录入菜单项\r\n")
        if self.get_user_role() != "admin":
            wx.MessageBox("您没有权限执行此操作", "权限错误", wx.OK | wx.ICON_ERROR)
            return
        if self.face_detector is None:
            wx.MessageBox(self.translate("模型未加载，无法进行人脸录入"), "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("error").format("模型未加载") + "\r\n")
            return
        if not self.new_register.IsEnabled():
            self.infoText.AppendText(self.getDateAndTime() + self.translate("正在进行其他操作，请先完成") + "\r\n")
            return
        self.new_register.Enable(False)
        self.finish_register.Enable(True)
        self.start_punchcard.Enable(False)
        self.end_punchcard.Enable(False)
        self.name = ""
        self.id = ID_WORKER_UNAVAILABLE
        self.pic_num = 0
        self.flag_registed = False
        self.loadDataBase(1)
        while self.id == ID_WORKER_UNAVAILABLE:
            id_input = wx.GetNumberFromUser(message=self.translate("请输入您的工号 (-1 不可用)"), 
                                            prompt=self.translate("工号: "), 
                                            caption=self.translate("新建录入"),
                                            value=ID_WORKER_UNAVAILABLE, parent=self, min=-1, max=100000000)
            if id_input == -1:
                wx.MessageBox(self.translate("工号 -1 不可用，请重新输入"), "警告", wx.OK | wx.ICON_WARNING)
                continue
            if id_input in self.knew_id:
                wx.MessageBox(self.translate("工号已存在，请重新输入"), "警告", wx.OK | wx.ICON_WARNING)
                self.id = ID_WORKER_UNAVAILABLE
                continue
            else:
                self.id = id_input
        while self.name == '':
            name_input = wx.GetTextFromUser(message=self.translate(f"请输入工号 {self.id} 对应的姓名"), 
                                            caption=self.translate("新建录入"),
                                            default_value="", parent=self)
            if not name_input:
                wx.MessageBox(self.translate("姓名不能为空"), "警告", wx.OK | wx.ICON_WARNING)
                continue
            name_dir_path = os.path.join(PATH_FACE, name_input)
            if os.path.exists(name_dir_path):
                wx.MessageBox(self.translate(f"姓名文件夹 '{name_input}' 已存在，请重新输入"), "警告", wx.OK | wx.ICON_WARNING)
                self.name = ''
                continue
            else:
                self.name = name_input
        os.makedirs(os.path.join(PATH_FACE, self.name))
        self.infoText.AppendText(self.getDateAndTime() + self.translate("register_start").format(self.id, self.name) + "\r\n")
        self.speak("请保持脸部正对摄像头，距离 30 到 50 厘米，确保光线充足", priority=True)
        self._is_finishing_register = False
        threading.Thread(target=self.register_cap, args=(event,)).start()

    def OnFinishRegisterClicked(self, event):
        self._is_finishing_register = True
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info("注册摄像头释放")
        self.cap = None
        self.new_register.Enable(True)
        self.finish_register.Enable(False)
        self.start_punchcard.Enable(True)
        self.end_punchcard.Enable(False)
        if hasattr(self, 'bmp') and self.bmp:
            self.bmp.SetBitmap(self.default_bitmap)
        self.infoText.AppendText(self.getDateAndTime() + self.translate("人脸录入已完成") + "\r\n")
        self.speak("人脸录入已完成", priority=True)
        self.loadDataBase(1)

    def punchcard_cap(self, event):
        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("start_punch") + "\r\n")
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("error").format("无法打开摄像头") + "\r\n")
            wx.CallAfter(self.OnEndPunchCard)
            return
        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("camera_open") + "\r\n")
        self.speak(self.translate("camera_open"), priority=True)

        last_punched_id = None
        last_punched_time = 0.0
        last_brightness_warning = 0
        last_log_update = 0
        last_bitmap_update = 0
        update_interval = 0.1
        frame_counter = 0

        while self.cap.isOpened():
            try:
                frame_start = time.time()
                flag, im_rd = self.cap.read()
                if not flag or im_rd is None:
                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("警告：摄像头读取失败，尝试重新读取...") + "\r\n")
                    time.sleep(0.1)
                    if not self.cap or not self.cap.isOpened():
                        break
                    continue
                im_rd = cv2.flip(im_rd, 1)
                img_rgb = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)

                current_time = time.time()
                if current_time - last_brightness_warning > 3:
                    if check_brightness(im_rd):
                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("lighting") + "\r\n")
                        self.speak(self.translate("lighting"))
                        last_brightness_warning = current_time

                frame_counter += 1
                biggest_face = None
                face_shape = None
                is_masked = False
                if frame_counter % 4 == 0:
                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    faces = self.face_detector(gray, 1)
                    if faces:
                        max_area = 0
                        for face in faces:
                            area = (face.right() - face.left()) * (face.bottom() - face.top())
                            if area > max_area:
                                max_area = area
                                biggest_face = face
                        if biggest_face:
                            self.last_detected_face = biggest_face
                            self.face_box_display_time = current_time
                            face_shape = self.shape_predictor(img_rgb, biggest_face)
                            try:
                                lower_points = [face_shape.part(i) for i in range(27, 36)] + [face_shape.part(i) for i in range(48, 68)]
                                invalid_points = sum(1 for p in lower_points if p.x <= 0 or p.y <= 0 or 
                                                    p.x >= img_rgb.shape[1] or p.y >= img_rgb.shape[0])
                                if invalid_points > len(lower_points) * 0.3:
                                    is_masked = True
                                else:
                                    face_img = img_rgb[biggest_face.top():biggest_face.bottom(), 
                                                      biggest_face.left():biggest_face.right()]
                                    if face_img.shape[0] > 0 and np.mean(face_img[face_img.shape[0]//2:]) < 60:
                                        is_masked = True
                                if is_masked:
                                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("mask_detected") + "\r\n")
                                    self.speak(self.translate("mask_detected"))
                            except Exception as e:
                                logging.warning(f"关键点检测失败: {e}")
                            self.face_lost_time = 0.0

                if self.last_detected_face and (current_time - self.face_box_display_time) < 3.0:
                    face = self.last_detected_face
                    rect_color = (255, 255, 0)
                    if self.liveness_state in [LIVENESS_STATE_PASSED, LIVENESS_STATE_PROCESSING]:
                        rect_color = (0, 255, 0)
                    elif self.liveness_state == LIVENESS_STATE_FAILED:
                        rect_color = (0, 0, 255)
                    cv2.rectangle(im_rd, (face.left(), face.top()), 
                                  (face.right(), face.bottom()), rect_color, 2)

                if biggest_face and (self.liveness_state in [LIVENESS_STATE_WAIT_FACE, LIVENESS_STATE_WAIT_BLINK]):
                    elapsed_time = time.time() - self.liveness_start_time
                    if elapsed_time > LIVENESS_STEP_TIMEOUT:
                        self.liveness_state = LIVENESS_STATE_FAILED
                        self.current_liveness_prompt = self.translate("timeout")
                        if current_time - last_log_update > 1:
                            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("timeout") + "\r\n")
                            last_log_update = current_time
                        self.speak(self.translate("timeout"))
                        self.liveness_state = LIVENESS_STATE_WAIT_FACE
                        self.liveness_start_time = time.time()
                        self.blink_count = 0
                        self.blink_consec_frames = 0
                        self.current_liveness_prompt = self.translate("adjust_camera")
                        if current_time - last_log_update > 1:
                            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("retry") + "\r\n")
                            last_log_update = current_time
                        self.speak(self.translate("retry"))
                    else:
                        try:
                            left_eye, right_eye = self.get_eye_points(face_shape)
                            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                            for point in left_eye + right_eye:
                                cv2.circle(im_rd, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
                            progress = min(100, int((self.blink_count / REQUIRED_BLINKS) * 100))
                            if hasattr(self, 'progress_bar') and self.progress_bar and self.progress_bar.IsShown():
                                wx.CallAfter(self.progress_bar.SetValue, progress)
                            if hasattr(self, 'blink_label') and self.blink_label and self.blink_label.IsShown():
                                wx.CallAfter(self.blink_label.SetLabel, self.translate("blink_progress").format(self.blink_count, REQUIRED_BLINKS))

                            if self.liveness_state == LIVENESS_STATE_WAIT_FACE:
                                self.liveness_state = LIVENESS_STATE_WAIT_BLINK
                                self.liveness_start_time = time.time()
                                self.blink_count = 0
                                self.blink_consec_frames = 0
                                self.current_liveness_prompt = self.translate("blink_prompt").format(REQUIRED_BLINKS)
                                if current_time - last_log_update > 1:
                                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("请进行活体检测: 眨眼") + "\r\n")
                                    last_log_update = current_time
                                self.speak(self.translate("blink_prompt").format(REQUIRED_BLINKS))
                            elif self.liveness_state == LIVENESS_STATE_WAIT_BLINK:
                                if ear < EAR_THRESHOLD:
                                    self.blink_consec_frames += 1
                                else:
                                    if self.blink_consec_frames >= EAR_CONSEC_FRAMES:
                                        self.blink_count += 1
                                        if current_time - last_log_update > 1:
                                            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate(f"检测到眨眼 {self.blink_count}/{REQUIRED_BLINKS}") + "\r\n")
                                            last_log_update = current_time
                                    self.blink_consec_frames = 0
                                if self.blink_count >= REQUIRED_BLINKS:
                                    self.liveness_state = LIVENESS_STATE_PASSED
                                    self.current_liveness_prompt = self.translate("success")
                                    if current_time - last_log_update > 1:
                                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("success") + "\r\n")
                                        last_log_update = current_time
                                    self.speak(self.translate("success"))
                        except Exception as e:
                            if current_time - last_log_update > 1:
                                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("error").format(f"活体检测失败 - {e}") + "\r\n")
                                last_log_update = current_time
                            self.liveness_state = LIVENESS_STATE_FAILED
                            self.current_liveness_prompt = self.translate("活体检测失败 (处理错误)")

                elif biggest_face and self.liveness_state == LIVENESS_STATE_PASSED:
                    self.liveness_state = LIVENESS_STATE_PROCESSING
                    self.current_liveness_prompt = self.translate("活体成功，正在识别人脸...")

                elif biggest_face and self.liveness_state == LIVENESS_STATE_PROCESSING:
                    self.current_liveness_prompt = self.translate("活体成功，正在识别人脸...")
                    if self.face_recognizer:
                        try:
                            current_face_feature = np.array(self.face_recognizer.compute_face_descriptor(img_rgb, face_shape))
                            recognized = False
                            worker_id = None
                            worker_name = None
                            if self.knew_id and self.knew_face_feature:
                                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + f"已加载注册用户数量: {len(self.knew_id)}\r\n")
                                for i in range(len(self.knew_id)):
                                    result = return_euclidean_distance(current_face_feature, self.knew_face_feature[i], is_masked)
                                    if result == "same":
                                        worker_id = self.knew_id[i]
                                        worker_name = self.knew_name[i]
                                        recognized = True
                                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + f"人脸匹配成功: 工号 {worker_id}, 姓名 {worker_name}\r\n")
                                        break
                                    else:
                                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + f"人脸匹配失败: 工号 {self.knew_id[i]}, 特征距离超出阈值\r\n")
                            else:
                                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("错误：未加载任何已注册用户") + "\r\n")
                                self.speak(self.translate("unrecognized"))
                                self.liveness_state = LIVENESS_STATE_WAIT_FACE
                                self.liveness_start_time = time.time()
                                self.blink_count = 0
                                self.blink_consec_frames = 0
                                self.current_liveness_prompt = self.translate("adjust_camera")
                                continue

                            if recognized:
                                current_time_float = time.time()
                                punchcard_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                if not self.checkWorkHours(punchcard_datetime):
                                    self.current_liveness_prompt = "签到时间不在上下班时间范围内，已忽略"
                                    if current_time - last_log_update > 1:
                                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.current_liveness_prompt + "\r\n")
                                        last_log_update = current_time
                                    self.speak(self.current_liveness_prompt)
                                elif last_punched_id == worker_id and (current_time_float - last_punched_time) < 60:
                                    self.current_liveness_prompt = self.translate("repeated_checkin")
                                    if current_time - last_log_update > 1:
                                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate(f"工号 {worker_id} 在 60 秒内重复签到，已忽略") + "\r\n")
                                        last_log_update = current_time
                                    self.speak(self.translate("repeated_checkin"))
                                else:
                                    photo_dir = os.path.join(PATH_PUNCHCARD_PHOTOS, str(worker_id))
                                    if not os.path.exists(photo_dir):
                                        os.makedirs(photo_dir)
                                    photo_path = os.path.join(photo_dir, f"{punchcard_datetime.replace(':', '-')}.jpg")
                                    cv2.imwrite(photo_path, im_rd)
                                    self.insertARow([worker_id, worker_name, punchcard_datetime, photo_path, "normal", "pending"], 2)
                                    last_punched_id = worker_id
                                    last_punched_time = current_time_float
                                    self.current_liveness_prompt = self.translate("punch_success").format(worker_name, worker_id)
                                    if current_time - last_log_update > 1:
                                        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.current_liveness_prompt + f" - {punchcard_datetime}\r\n")
                                        wx.CallAfter(wx.MessageBox, self.current_liveness_prompt, "签到成功", wx.OK | wx.ICON_INFORMATION)
                                        last_log_update = current_time
                                    self.speak(self.translate("punch_success").format(worker_name, worker_id), priority=True)
                                    time.sleep(3)
                                self.liveness_state = LIVENESS_STATE_WAIT_FACE
                                self.liveness_start_time = time.time()
                                self.blink_count = 0
                                self.blink_consec_frames = 0
                                self.current_liveness_prompt = self.translate("adjust_camera")
                            else:
                                self.current_liveness_prompt = self.translate("unrecognized")
                                if current_time - last_log_update > 1:
                                    wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("错误：人脸未注册") + "\r\n")
                                    last_log_update = current_time
                                self.speak(self.translate("unrecognized"))
                                self.liveness_state = LIVENESS_STATE_WAIT_FACE
                                self.liveness_start_time = time.time()
                                self.blink_count = 0
                                self.blink_consec_frames = 0
                                self.current_liveness_prompt = self.translate("adjust_camera")
                        except Exception as e:
                            if current_time - last_log_update > 1:
                                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("error").format(f"刷脸识别失败 - {e}") + "\r\n")
                                last_log_update = current_time
                            self.current_liveness_prompt = self.translate("识别处理错误")
                    else:
                        self.current_liveness_prompt = self.translate("model_error")
                        if current_time - last_log_update > 1:
                            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("error").format("无法签到") + "\r\n")
                            last_log_update = current_time
                        wx.CallAfter(self.OnEndPunchCard)
                        return

                elif biggest_face and self.liveness_state == LIVENESS_STATE_FAILED:
                    self.current_liveness_prompt = self.current_liveness_prompt
                    if (time.time() - self.liveness_start_time) > 3:
                        self.liveness_state = LIVENESS_STATE_WAIT_FACE
                        self.liveness_start_time = time.time()
                        self.blink_count = 0
                        self.blink_consec_frames = 0
                        self.current_liveness_prompt = self.translate("adjust_camera")
                        if current_time - last_log_update > 1:
                            wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("活体检测失败，请重试") + "\r\n")
                            last_log_update = current_time
                        self.speak("活体检测失败，请重试")

                elif not biggest_face and (self.liveness_state in [LIVENESS_STATE_WAIT_FACE, LIVENESS_STATE_WAIT_BLINK, LIVENESS_STATE_PASSED, LIVENESS_STATE_PROCESSING]):
                    if self.face_lost_time == 0.0:
                        self.face_lost_time = time.time()
                    elif (time.time() - self.face_lost_time) > 2.0:
                        if self.liveness_state != LIVENESS_STATE_WAIT_FACE:
                            self.liveness_state = LIVENESS_STATE_WAIT_FACE
                            self.liveness_start_time = time.time()
                            self.blink_count = 0
                            self.blink_consec_frames = 0
                            self.current_liveness_prompt = self.translate("face_lost")
                            if current_time - last_log_update > 1:
                                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("face_lost") + "\r\n")
                                last_log_update = current_time
                            self.speak(self.translate("face_lost"))
                        self.current_liveness_prompt = self.translate("face_lost")

                if current_time - last_bitmap_update >= update_interval:
                    im_rd = cv2AddChineseText(im_rd, self.current_liveness_prompt, (10, 30), textColor=(255, 255, 0), textSize=40)
                    buf = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB).tobytes()
                    w, h = im_rd.shape[1], im_rd.shape[0]
                    img = wx.Image(w, h, buf)
                    wx_bitmap = wx.Bitmap(img)
                    if hasattr(self, 'bmp') and self.bmp and self.bmp.IsShown():
                        wx.CallAfter(self.bmp.SetBitmap, wx_bitmap)
                    last_bitmap_update = current_time

                frame_time = time.time() - frame_start
                sleep_time = max(0, (1.0 / FRAME_RATE) - frame_time)
                time.sleep(sleep_time)
            except Exception as e:
                wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("error").format(f"摄像头线程异常 - {e}") + "\r\n")
                traceback.print_exc()
                break

        wx.CallAfter(self.infoText.AppendText, self.getDateAndTime() + self.translate("摄像头线程结束") + "\r\n")
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        if not self._is_finishing_punchcard:
            wx.CallAfter(self.OnEndPunchCard)

    def OnStartPunchCardClicked(self, event):
        if self.face_detector is None:
            wx.MessageBox(self.translate("模型未加载，无法进行刷脸签到"), "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("error").format("模型未加载") + "\r\n")
            return
        if not self.start_punchcard.IsEnabled():
            self.infoText.AppendText(self.getDateAndTime() + self.translate("正在进行其他操作，请先完成") + "\r\n")
            return
        self.start_punchcard.Enable(False)
        self.end_punchcard.Enable(True)
        if self.user_role == "admin":
            self.new_register.Enable(False)
            self.finish_register.Enable(False)
            self.set_work_hours.Enable(False)
            self.makeup_punchcard.Enable(False)
        self.liveness_state = LIVENESS_STATE_WAIT_FACE
        self.liveness_start_time = time.time()
        self.blink_count = 0
        self.blink_consec_frames = 0
        self.current_liveness_prompt = self.translate("adjust_camera")
        self._is_finishing_punchcard = False
        threading.Thread(target=self.punchcard_cap, args=(event,)).start()

    def OnEndPunchCard(self, event=None):
        self._is_finishing_punchcard = True
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info("签到摄像头释放")
        self.cap = None
        self.start_punchcard.Enable(True)
        self.end_punchcard.Enable(False)
        if self.user_role == "admin":
            self.new_register.Enable(True)
            self.finish_register.Enable(False)
            self.set_work_hours.Enable(True)
            self.makeup_punchcard.Enable(True)
        if hasattr(self, 'bmp') and self.bmp:
            self.bmp.SetBitmap(self.default_bitmap)
        self.liveness_state = LIVENESS_STATE_IDLE
        self.current_liveness_prompt = ""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.SetValue(0)
        if hasattr(self, 'blink_label') and self.blink_label:
            self.blink_label.SetLabel(self.translate("blink_progress").format(0, REQUIRED_BLINKS))
        self.infoText.AppendText(self.getDateAndTime() + self.translate("签到已结束") + "\r\n")

    def OnOpenLogcatClicked(self, event):
        if self.grid_logcat and self.grid_logcat.IsShown():
            self.infoText.AppendText(self.getDateAndTime() + "考勤日志窗口已打开\r\n")
            return
        self.grid_logcat = wx.Frame(self, title="考勤日志", size=(800, 400))
        self.grid_logcat.Bind(wx.EVT_CLOSE, self.OnCloseLogcatClicked)
        panel = wx.Panel(self.grid_logcat)
        self.logcat_grid = wx.grid.Grid(panel, pos=(10, 10), size=(780, 340))
        self.logcat_grid.CreateGrid(0, 6)
        self.logcat_grid.SetColLabelValue(0, "日志ID")
        self.logcat_grid.SetColLabelValue(1, "工号")
        self.logcat_grid.SetColLabelValue(2, "姓名")
        self.logcat_grid.SetColLabelValue(3, "签到时间")
        self.logcat_grid.SetColLabelValue(4, "照片路径")
        self.logcat_grid.SetColLabelValue(5, "签到类型")
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            if self.user_role == "admin":
                cursor.execute("SELECT log_id, worker_id, worker_name, punchcard_datetime, photo_path, punch_type FROM logcat")
            else:
                cursor.execute("SELECT log_id, worker_id, worker_name, punchcard_datetime, photo_path, punch_type FROM logcat WHERE username = ?", (self.current_username,))
            rows = cursor.fetchall()
            self.logcat_grid.AppendRows(len(rows))
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    self.logcat_grid.SetCellValue(i, j, str(value))  # 直接使用数据库中的中文值
            conn.close()
            self.infoText.AppendText(self.getDateAndTime() + f"加载了 {len(rows)} 条考勤日志\r\n")
        except Exception as e:
            logging.error(f"加载考勤日志失败: {e}")
            wx.MessageBox(f"加载考勤日志失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + f"加载考勤日志失败: {e}\r\n")
        self.grid_logcat.Show()
    def OnCloseLogcatClicked(self, event):
        if self.grid_logcat:
            self.grid_logcat.Hide()
            self.infoText.AppendText(self.getDateAndTime() + "考勤日志窗口已关闭\r\n")

    def OnExportLogcatClicked(self, event):
        with wx.FileDialog(self, "导出考勤日志", wildcard="Excel 文件 (*.xlsx)|*.xlsx",
                        style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path = fileDialog.GetPath()
            try:
                conn = sqlite3.connect('attendance.db')
                if self.user_role == "admin":
                    df = pd.read_sql_query("SELECT log_id, worker_id, worker_name, punchcard_datetime, photo_path, punch_type FROM logcat", conn)
                else:
                    df = pd.read_sql_query(
                        "SELECT log_id, worker_id, worker_name, punchcard_datetime, photo_path, punch_type FROM logcat WHERE username = ?",
                        conn,
                        params=(self.current_username,)
                    )
                conn.close()

                # 定义签到类型的中英映射
                PUNCH_TYPE_MAP = {
                    "normal": "正常打卡",
                    "early": "早到打卡",
                    "late": "迟到打卡",
                    "admin": "管理员打卡"
                }

                # 将 punch_type 列的英文值转换为中文
                df['punch_type'] = df['punch_type'].map(lambda x: PUNCH_TYPE_MAP.get(x, x))

                # 设置中文列名
                df.columns = ["日志ID", "工号", "姓名", "签到时间", "照片路径", "签到类型"]

                df.to_excel(path, index=False, engine='openpyxl')  # 导出为 .xlsx 文件
                self.infoText.AppendText(self.getDateAndTime() + f"考勤日志已导出到 {path}\r\n")
            except Exception as e:
                logging.error(f"导出考勤日志失败: {e}")
                wx.MessageBox(f"导出考勤日志失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
                self.infoText.AppendText(self.getDateAndTime() + f"导出考勤日志失败: {e}\r\n")

    def OnOpenDashboardClicked(self, event):
        if self.dashboard and self.dashboard.IsShown():
            self.infoText.AppendText(self.getDateAndTime() + "数据仪表板已打开\r\n")
            return
        self.dashboard = wx.Frame(self, title="数据仪表板", size=(1000, 800))
        self.dashboard.Bind(wx.EVT_CLOSE, lambda evt: self.dashboard.Hide())
        panel = wx.Panel(self.dashboard)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        canvas = FigureCanvasWxAgg(panel, -1, fig)
        canvas.SetPosition((10, 10))
        canvas.SetSize((980, 780))

        try:
            conn = sqlite3.connect('attendance.db')
            if self.user_role == "admin":
                df = pd.read_sql_query("SELECT worker_id, worker_name, punchcard_datetime FROM logcat", conn)
            else:
                df = pd.read_sql_query("SELECT worker_id, worker_name, punchcard_datetime FROM logcat WHERE username = ?", conn, params=(self.current_username,))
            conn.close()

            # 每日签到人数图表
            if not df.empty:
                df['punchcard_datetime'] = pd.to_datetime(df['punchcard_datetime'])
                df['date'] = df['punchcard_datetime'].dt.date
                df_first_punch = df.groupby(['worker_id', 'date']).first().reset_index()
                daily_counts = df_first_punch.groupby('date').size()

                ax1.plot(daily_counts.index, daily_counts.values, marker='o', label='签到人数')
                ax1.set_title("每日签到人数")
                ax1.set_xlabel("日期")
                ax1.set_ylabel("人数")
                ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # 确保签到人数为整数
                ax1.grid(True)
                ax1.legend()
                ax1.tick_params(axis='x', rotation=0, labelsize=8)  # 日期水平显示
                ax1.set_xticks(range(len(daily_counts.index)))
                ax1.set_xticklabels(daily_counts.index, rotation=0, ha='center')
                ax1.margins(x=0.05)
            else:
                ax1.text(0.5, 0.5, "暂无签到数据", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
                ax1.set_title("每日签到人数")
                ax1.set_xlabel("日期")
                ax1.set_ylabel("人数")

            # 员工签到次数统计图表
            if not df.empty:
                worker_counts = df_first_punch.groupby('worker_name').size().sort_values(ascending=False)
                ax2.bar(range(len(worker_counts)), worker_counts.values, color='skyblue')
                ax2.set_title("员工签到次数统计")
                ax2.set_xlabel("员工姓名")
                ax2.set_ylabel("签到次数")
                ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # 确保签到次数为整数
                ax2.grid(True, axis='y')
                ax2.set_xticks(range(len(worker_counts)))  # 显示所有员工
                ax2.set_xticklabels(worker_counts.index, rotation=0, ha='center', fontsize=8)  # 姓名水平显示，完全显示
                ax2.margins(x=0.05)  # 增加 X 轴间距，避免姓名重叠
            else:
                ax2.text(0.5, 0.5, "暂无签到数据", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
                ax2.set_title("员工签到次数统计")
                ax2.set_xlabel("员工姓名")
                ax2.set_ylabel("签到次数")

            plt.tight_layout()
            canvas.draw()
            self.infoText.AppendText(self.getDateAndTime() + "数据仪表板已加载\r\n")
        except Exception as e:
            logging.error(f"加载数据仪表板失败: {e}")
            wx.MessageBox(f"加载数据仪表板失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + f"加载数据仪表板失败: {e}\r\n")
        self.dashboard.Show()

    def OnManageUsersClicked(self, event):
        if self.get_user_role() != "admin":
            wx.MessageBox("您没有权限执行此操作", "权限错误", wx.OK | wx.ICON_ERROR)
            return
        self.user_management = wx.Frame(self, title="用户管理", size=(600, 400))
        self.user_management.Bind(wx.EVT_CLOSE, lambda evt: self.user_management.Destroy())
        panel = wx.Panel(self.user_management)
        self.user_grid = wx.grid.Grid(panel, pos=(10, 10), size=(580, 300))
        self.user_grid.CreateGrid(0, 4)
        self.user_grid.SetColLabelValue(0, "用户ID")
        self.user_grid.SetColLabelValue(1, "用户名")
        self.user_grid.SetColLabelValue(2, "密码")
        self.user_grid.SetColLabelValue(3, "角色")
        add_button = wx.Button(panel, label="添加用户", pos=(10, 320))
        edit_button = wx.Button(panel, label="编辑用户", pos=(120, 320))
        delete_button = wx.Button(panel, label="删除用户", pos=(230, 320))
        add_button.Bind(wx.EVT_BUTTON, self.OnAddUserClicked)
        edit_button.Bind(wx.EVT_BUTTON, self.OnEditUserClicked)
        delete_button.Bind(wx.EVT_BUTTON, self.OnDeleteUserClicked)
        self.loadUserData()
        self.user_management.Show()

    def loadUserData(self):
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, username, password, role FROM users")
            rows = cursor.fetchall()
            num_rows = self.user_grid.GetNumberRows()
            if num_rows > 0:
                self.user_grid.DeleteRows(0, num_rows, True)
            self.user_grid.AppendRows(len(rows))
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    self.user_grid.SetCellValue(i, j, str(value))
            conn.close()
            self.infoText.AppendText(self.getDateAndTime() + f"加载了 {len(rows)} 条用户数据\r\n")
        except Exception as e:
            logging.error(f"加载用户数据失败: {e}")
            wx.MessageBox(f"加载用户数据失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + f"加载用户数据失败: {e}\r\n")

    def OnAddUserClicked(self, event):
        user_id = wx.GetNumberFromUser("请输入用户ID", "用户ID:", "添加用户", -1, -1, 1000000, self.user_management)
        if user_id == -1:
            return
        username = wx.GetTextFromUser("请输入用户名", "添加用户", "", self.user_management)
        if not username:
            return
        password = wx.GetTextFromUser("请输入密码", "添加用户", "", self.user_management)
        if not password:
            return
        role = wx.GetSingleChoice("请选择角色", "添加用户", ["admin", "user"], self.user_management)
        if not role:
            return
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (user_id, username, password, role) VALUES (?, ?, ?, ?)", 
                           (user_id, username, password, role))
            conn.commit()
            conn.close()
            self.loadUserData()
            self.infoText.AppendText(self.getDateAndTime() + f"用户 {username} 添加成功\r\n")
        except Exception as e:
            logging.error(f"添加用户失败: {e}")
            wx.MessageBox(f"添加用户失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + f"添加用户失败: {e}\r\n")

    def OnEditUserClicked(self, event):
        selected_row = self.user_grid.GetSelectedRows()
        if not selected_row:
            wx.MessageBox("请选择一个用户进行编辑", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        row = selected_row[0]
        user_id = int(self.user_grid.GetCellValue(row, 0))
        username = wx.GetTextFromUser("请输入新用户名", "编辑用户", self.user_grid.GetCellValue(row, 1), self.user_management)
        if not username:
            return
        password = wx.GetTextFromUser("请输入新密码", "编辑用户", self.user_grid.GetCellValue(row, 2), self.user_management)
        if not password:
            return
        role = wx.GetSingleChoice("请选择新角色", "编辑用户", ["admin", "user"], self.user_management, initialSelection=0 if self.user_grid.GetCellValue(row, 3) == "admin" else 1)
        if not role:
            return
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET username = ?, password = ?, role = ? WHERE user_id = ?", 
                           (username, password, role, user_id))
            conn.commit()
            conn.close()
            self.loadUserData()
            self.infoText.AppendText(self.getDateAndTime() + f"用户 {username} 编辑成功\r\n")
        except Exception as e:
            logging.error(f"编辑用户失败: {e}")
            wx.MessageBox(f"编辑用户失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + f"编辑用户失败: {e}\r\n")

    def OnDeleteUserClicked(self, event):
        selected_row = self.user_grid.GetSelectedRows()
        if not selected_row:
            wx.MessageBox("请选择一个用户进行删除", "提示", wx.OK | wx.ICON_INFORMATION)
            return
        row = selected_row[0]
        user_id = int(self.user_grid.GetCellValue(row, 0))
        username = self.user_grid.GetCellValue(row, 1)
        if username == "admin":
            wx.MessageBox("不能删除默认管理员用户", "错误", wx.OK | wx.ICON_ERROR)
            return
        if wx.MessageBox(f"确定要删除用户 {username} 吗？", "确认删除", wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
            return
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            self.loadUserData()
            self.infoText.AppendText(self.getDateAndTime() + f"用户 {username} 删除成功\r\n")
        except Exception as e:
            logging.error(f"删除用户失败: {e}")
            wx.MessageBox(f"删除用户失败: {e}", "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + f"删除用户失败: {e}\r\n")

    def OnSetWorkHoursClicked(self, event):
        if self.get_user_role() != "admin":
            wx.MessageBox(self.translate("no_admin_priv"), "权限错误", wx.OK | wx.ICON_ERROR)
            return

        dlg = wx.TextEntryDialog(self, "请输入上班时间 (HH:MM，例如 09:00)", self.translate("set_work_hours"), self.work_start_time)
        if dlg.ShowModal() == wx.ID_OK:
            start_time = dlg.GetValue()
            dlg2 = wx.TextEntryDialog(self, "请输入下班时间 (HH:MM，例如 18:00)", self.translate("set_work_hours"), self.work_end_time)
            if dlg2.ShowModal() == wx.ID_OK:
                end_time = dlg2.GetValue()
                try:
                    # 校验时间格式
                    start_dt = datetime.datetime.strptime(start_time, '%H:%M')
                    end_dt = datetime.datetime.strptime(end_time, '%H:%M')

                    if start_dt >= end_dt:
                        raise ValueError("下班时间必须晚于上班时间")

                    conn = sqlite3.connect('attendance.db')
                    cursor = conn.cursor()

                    # 尝试更新已有记录
                    cursor.execute("UPDATE work_hours SET start_time = ?, end_time = ? WHERE username = ?", 
                                (start_time, end_time, self.current_username))

                    # 如果没有更新到任何记录，则插入新记录（不指定 id，让其自动递增）
                    if cursor.rowcount == 0:
                        cursor.execute("INSERT INTO work_hours (start_time, end_time, username) VALUES (?, ?, ?)", 
                                    (start_time, end_time, self.current_username))

                    conn.commit()
                    conn.close()

                    self.work_start_time = start_time
                    self.work_end_time = end_time
                    msg = self.translate("work_hours_set").format(start_time, end_time)
                    self.infoText.AppendText(self.getDateAndTime() + msg + "\r\n")
                    self.speak(msg)

                except ValueError as ve:
                    wx.MessageBox(f"错误：{str(ve)}", "错误", wx.OK | wx.ICON_ERROR)
                    self.infoText.AppendText(self.getDateAndTime() + f"错误：{str(ve)}\r\n")
                except Exception as e:
                    logging.error(f"更新上下班时间失败: {e}")
                    wx.MessageBox(f"错误：更新上下班时间失败 - {e}", "错误", wx.OK | wx.ICON_ERROR)
                    self.infoText.AppendText(self.getDateAndTime() + f"错误：更新上下班时间失败 - {e}\r\n")
            dlg2.Destroy()
        dlg.Destroy()


    def checkWorkHours(self, punch_time):
        try:
            punch_datetime = datetime.datetime.strptime(punch_time, "%Y-%m-%d %H:%M:%S")
            punch_time_str = punch_datetime.strftime("%H:%M")
            start_dt = datetime.datetime.strptime(self.work_start_time, "%H:%M")
            end_dt = datetime.datetime.strptime(self.work_end_time, "%H:%M")
            punch_dt = datetime.datetime.strptime(punch_time_str, "%H:%M")
            return start_dt <= punch_dt <= end_dt
        except ValueError as e:
            logging.error(f"检查上下班时间失败: {e}")
            return False

    def OnMakeupPunchCard(self, event):
        if self.get_user_role() != "admin":
            wx.MessageBox(self.translate("no_admin_priv"), "权限错误", wx.OK | wx.ICON_ERROR)
            return

        # Step 1: Prompt for worker ID
        worker_id = wx.GetNumberFromUser(
            message=self.translate("请输入需要补卡的工号"),
            prompt=self.translate("工号: "),
            caption=self.translate("补卡功能"),
            value=ID_WORKER_UNAVAILABLE,
            parent=self,
            min=-1,
            max=100000000
        )
        if worker_id == -1:
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：工号 -1 不可用") + "\r\n")
            return

        # Step 2: Verify if the worker exists in the database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM worker_info WHERE id = ? AND (username = ? OR username IS NULL)", 
                       (worker_id, self.current_username))
        result = cursor.fetchone()
        if not result:
            conn.close()
            wx.MessageBox(self.translate("错误：工号 {} 未注册").format(worker_id), "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：工号 {} 未注册").format(worker_id) + "\r\n")
            return
        worker_name = result[0]

        # Step 3: Prompt for punch card date and time
        punch_date = wx.GetTextFromUser(
            message=self.translate("请输入补卡日期 (格式: YYYY-MM-DD，例如 2025-05-02)"),
            caption=self.translate("补卡功能"),
            default_value=datetime.datetime.now().strftime("%Y-%m-%d"),
            parent=self
        )
        if not punch_date:
            conn.close()
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：补卡日期不能为空") + "\r\n")
            return

        punch_time = wx.GetTextFromUser(
            message=self.translate("请输入补卡时间 (格式: HH:MM:SS，例如 09:00:00)"),
            caption=self.translate("补卡功能"),
            default_value=datetime.datetime.now().strftime("%H:%M:%S"),
            parent=self
        )
        if not punch_time:
            conn.close()
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：补卡时间不能为空") + "\r\n")
            return

        # Step 4: Validate date and time format
        try:
            punchcard_datetime = f"{punch_date} {punch_time}"
            datetime.datetime.strptime(punchcard_datetime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            conn.close()
            wx.MessageBox(self.translate("错误：日期或时间格式错误，请使用 YYYY-MM-DD HH:MM:SS 格式"), 
                          "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：日期或时间格式错误") + "\r\n")
            return

        # Step 5: Check if the punch time is within work hours
        if not self.checkWorkHours(punchcard_datetime):
            conn.close()
            wx.MessageBox(self.translate("错误：补卡时间不在上下班时间范围内"), "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：补卡时间不在上下班时间范围内") + "\r\n")
            return

        # Step 6: Check for duplicate punch within 60 seconds (optional, depending on your requirements)
        cursor.execute("""
            SELECT punchcard_datetime FROM logcat 
            WHERE worker_id = ? AND punchcard_datetime >= ? AND punchcard_datetime <= ?
        """, (worker_id, 
              (datetime.datetime.strptime(punchcard_datetime, "%Y-%m-%d %H:%M:%S") - datetime.timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S"),
              (datetime.datetime.strptime(punchcard_datetime, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")))
        if cursor.fetchone():
            conn.close()
            wx.MessageBox(self.translate("错误：该时间段内已有签到记录，请勿重复补卡"), "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：工号 {} 在该时间段内已有签到记录").format(worker_id) + "\r\n")
            return

        # Step 7: Insert the makeup punch card record (no photo for makeup punch)
        try:
            self.insertARow([worker_id, worker_name, punchcard_datetime, "", "makeup", "completed"], 2)
            conn.close()
            wx.MessageBox(self.translate("补卡成功: {} (工号: {})").format(worker_name, worker_id), 
                          "成功", wx.OK | wx.ICON_INFORMATION)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("补卡成功: {} (工号: {}) - {}").format(worker_name, worker_id, punchcard_datetime) + "\r\n")
            self.speak(self.translate("补卡成功: {} (工号: {})").format(worker_name, worker_id), priority=True)
        except Exception as e:
            conn.close()
            logging.error(f"补卡失败: {e}")
            wx.MessageBox(self.translate("错误：补卡失败 - {}").format(e), "错误", wx.OK | wx.ICON_ERROR)
            self.infoText.AppendText(self.getDateAndTime() + self.translate("错误：补卡失败 - {}").format(e) + "\r\n")

    def OnExit(self, event):
        """处理窗口关闭事件，清理资源。"""
        self.infoText.AppendText(self.getDateAndTime() + "正在关闭员工考勤系统...\r\n")
        logging.info("应用程序正在关闭")

        # 停止颜色变换定时器
        if hasattr(self, 'color_timer') and self.color_timer.IsRunning():
            self.color_timer.Stop()

        # 释放摄像头
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info("摄像头已释放")

        # 停止语音线程和引擎
        if hasattr(self, 'speech_queue'):
            try:
                self.speech_queue.put(None)  # 通知语音线程停止
                if hasattr(self, 'speech_thread') and self.speech_thread.is_alive():
                    self.speech_thread.join(timeout=2.0)
                self.speech_engine.stop()
                self.speech_engine.endLoop()  # 确保语音引擎正确关闭
                logging.info("语音引擎已关闭")
            except Exception as e:
                logging.error(f"关闭语音引擎失败: {e}")

        # 关闭所有打开的子窗口（如日志窗口、仪表板等）
        if hasattr(self, 'grid_logcat') and self.grid_logcat and self.grid_logcat.IsShown():
            self.grid_logcat.Destroy()
        if hasattr(self, 'dashboard') and self.dashboard and self.dashboard.IsShown():
            self.dashboard.Destroy()
        if hasattr(self, 'user_management') and self.user_management and self.user_management.IsShown():
            self.user_management.Destroy()

        # 记录关闭日志
        self.infoText.AppendText(self.getDateAndTime() + "员工考勤系统已关闭\r\n")
        logging.info("应用程序已关闭")

        # 销毁主窗口，退出程序
        self.Destroy()

# 程序入口
if __name__ == '__main__':
    try:
        app = wx.App(False)  # 创建 wxPython 应用程序实例
        frame = WAS()  # 创建 WAS 主窗口实例
        frame.Show(True)  # 显示主窗口
        app.MainLoop()  # 启动应用程序主事件循环
    except Exception as e:
        logging.error(f"应用程序启动失败: {e}")
        print(f"错误: 应用程序启动失败 - {e}")