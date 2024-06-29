import cv2
import numpy as np
import time
from ye_agv import MyAgv

SIGNAL = 1
LANE = 2
PAUSE = 3
STOP = 4
mode = LANE
MA = MyAgv('/dev/ttyAMA2', 115200)

def cam_init():
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라 장치를 의미, 만약 다른 장치라면 숫자를 조정하여야 함
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오를 읽을 수 없습니다. 종료합니다.")
            break

        # 화면 크기 조정 (640x480)
        frame = cv2.resize(frame, (640, 480))
        
        yield frame
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 시 캡처 객체와 창 닫기
    cap.release()
    cv2.destroyAllWindows()

def check_signal_start():
    global mode
    for frame in cam_init():
        # ROI 설정 [0:200, 320:640]
        x_start = 250
        x_end = 400
        y_start = 180
        y_end = 230

        roi = frame[y_start:y_end, x_start:x_end]

        # ROI를 HSV 색 공간으로 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 빨간색의 HSV 범위 설정 (0-10도, 170-180도)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 빨간색만 검출
        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        # 이진화된 이미지에서 점 개수 세기
        num_points = cv2.countNonZero(mask_red)

        # 점 개수가 70개 이상인 경우 동작 수행
        if num_points >= 50:
            print("50개 이상의 점이 감지되었습니다. 동작을 수행합니다.")
            mode = LANE
            break

        # 화면에 원본 이미지 및 ROI 표시
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        # cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)

def check_stopline():
    global mode
    for frame in cam_init():
        # ROI 설정 [0:200, 320:640]
        x_start = 100
        x_end = 540
        y_start = 350
        y_end = 480

        roi = frame[y_start:y_end, x_start:x_end]

        # ROI를 HSV 색 공간으로 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 노란색의 HSV 범위 설정 (20-30도)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # 노란색만 검출
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

        # 이진화된 이미지에서 점 개수 세기
        num_points = cv2.countNonZero(mask_yellow)

        # 점 개수가 300개 이상인 경우 동작 수행
        if num_points >= 300:
            print("300개 이상의 점이 감지되었습니다. 동작을 수행합니다.")
            mode = SIGNAL
            return True

        # 화면에 원본 이미지 및 ROI 표시
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        # cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)

        return False  
    
def process_frame():
    for frame in cam_init():
         # ROI 설정 [0:200, 320:640]
        x_start = 0
        x_end = 640
        y_start = 350
        y_end = 480

        roi = frame[y_start:y_end, x_start:x_end]

        # 프레임을 그레이스케일로 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 이진화 (바이너리 이미지로 변환)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # 프레임을 왼쪽과 오른쪽으로 나누기
        height, width = binary.shape
        left_region = binary[:, :width//2]
        right_region = binary[:, width//2:]
        
        # 왼쪽과 오른쪽 영역의 점 개수 계산
        left_count = np.sum(left_region == 255)
        right_count = np.sum(right_region == 255)

        # cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)
        
        return left_count, right_count
    
def line_tracing():
    global mode
    left_count, right_count = process_frame()
    
    if abs(left_count - right_count) >= 9000:
        speed = 10
        kp = int(abs(left_count - right_count)*0.0055)
        if kp >= 40:
            kp = 40
        if kp <= 30:
            kp = 30
        if left_count > right_count:
            MA.turn_left(speed, kp, 1)
        else:
            MA.turn_right(speed, int(kp*0.3), 1)
    elif left_count >= 1000 and right_count <=1000:
        MA.counterclockwise_rotation(5, 1)
    else:
        MA.go_ahead(10,1)

    if check_stopline() == True:
        time.sleep(1)
    if check_crosswalk() == True:
        time.sleep(1)

def pause():
    global mode
    time.sleep(5)
    MA.go_ahead(10,3)
    mode = LANE

def check_crosswalk():
    global mode
    for frame in cam_init():
        # ROI 설정 [0:200, 320:640]
        x_start = 120
        x_end = 520
        y_start = 300
        y_end = 480

        roi = frame[y_start:y_end, x_start:x_end]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 이진화 (바이너리 이미지로 변환)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # 흰색 점 개수 계산
        num_points = np.sum(gray == 255)

        # 점 개수가 300개 이상인 경우 동작 수행
        if num_points >= 10000:
            print("10000개 이상의 점이 감지되었습니다. 동작을 수행합니다.")
            mode = PAUSE
            return True

        # 화면에 원본 이미지 및 ROI 표시
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        # cv2.imshow('Original', frame)
        cv2.imshow('ROI', roi)

        return False   
    
def stop():
    MA.stop()

if __name__ == "__main__":
    while True:
        while mode == SIGNAL:
            check_signal_start()
        time.sleep(1)
        while mode == LANE:
            line_tracing()
        while mode == STOP:
            stop()
        while mode == PAUSE:
            pause()
