from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
# from typing import TypedDict, NotRequired
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
# track-doorをインポート
import track_door

history = []

# ヘッドレス環境かどうかを判定（GUI機能が使えるかチェック）
def is_headless_environment():
    """ヘッドレス環境（GUI機能が無い）かどうかを判定"""
    return os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or not os.environ.get('DISPLAY')


# YOLO11モデルを読み込む
model = YOLO("yolo11n.pt")

# 
# model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)

# 動画ファイルを開く
video_path = "videos/test2.mp4"
video_filename = video_path.split("/")[-1].split(".")[0]
cap = cv2.VideoCapture(video_path)

# トラッキング履歴と状態
track_history = defaultdict(lambda: [])
paused = False
exit_requested = False

# 入退室判定用ゾーン（左上と右下をマウスで指定）
zone_rect = None  # (x1, y1, x2, y2)
zone_points = []
zone_set = False

# 各IDの最後の中心座標と入退室ラベル
first_positions = {}
last_positions = {}
all_person_ids = set()
active_ids_prev = set()

# 各person IDについて、最初に検出されたフレームのマスク画像を保持する
first_masks = {}
# 各person IDについて、最初に検出されたときのフレームのカラー画像を保持する
first_frames = {}

# 結果画像を出力するために，最後のフレーム画像を保持する。
last_frame = None


cmap = plt.get_cmap("tab20")

def id_to_color(id_value):
    r, g, b, _ = cmap(id_value % cmap.N)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def is_inside_zone(pos):
    if zone_rect is None:
        return False
    x1, y1, x2, y2 = zone_rect
    x, y = pos
    return x1 <= x <= x2 and y1 <= y <= y2


def mouse_callback(event, x, y, flags, param):
    global zone_points, zone_rect, zone_set
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))
        if len(zone_points) == 2:
            (x1, y1), (x2, y2) = zone_points
            zone_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            zone_set = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右クリックでリセット
        zone_points = []
        zone_rect = None
        zone_set = False


# 最初のフレームを取得し、ゾーン確定まで一時停止
success, frame = cap.read()
if not success:
    cap.release()
    raise SystemExit("動画の読み込みに失敗しました")

# ヘッドレス環境への対応
HEADLESS = is_headless_environment()

if not HEADLESS:
    # GUI利用可能な環境：マウスでゾーン指定
    cv2.namedWindow("YOLO11 Tracking")
    cv2.setMouseCallback("YOLO11 Tracking", mouse_callback)
    
    # ゾーン確定フェーズ（1フレーム目で停止）
    while not zone_set and not exit_requested:
        preview = frame.copy()
        preview = track_door.detect_door(preview)
        # 指定済みの点を表示
        for pt in zone_points:
            cv2.circle(preview, pt, 4, (0, 255, 255), -1)
        # 2点揃ったら矩形を描画
        if len(zone_points) == 2:
            (x1, y1), (x2, y2) = zone_points
            cv2.rectangle(
                preview,
                (min(x1, x2), min(y1, y2)),
                (max(x1, x2), max(y1, y2)),
                (255, 0, 0),
                2,
            )
        cv2.putText(
            preview,
            "add 2points with left click (reset with right click)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "start process after definition / q to end process",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("YOLO11 Tracking", preview)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            exit_requested = True
            break
else:
    # ヘッドレス環境：デフォルトゾーンを使用
    print("Headless environment detected. Using default zone.", file=sys.stderr)
    print("If you want to specify a custom zone, set zone_rect in the code.", file=sys.stderr)
    # デフォルトゾーン設定：フレームの中央部分を使用（例：x=100-500, y=100-400）
    h, w = frame.shape[:2]
    # 例：フレームの20%-80%の領域をゾーンとする
    zone_rect = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
    zone_set = True
    print(f"Default zone set: {zone_rect}", file=sys.stderr)

# zone_rectに選択したボックスの座標が記録されている

# ユーザが終了を要求していた場合はプログラムを終了
if exit_requested:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# 正常に動画フレームを読み込める限りループ処理
while cap.isOpened():
    # ポーズ中は新規フレームを読まない
    if not paused:
        success, frame = cap.read()
    else:
        success = True  # ポーズ中は前フレームをそのまま表示

    if not success:
        # 動画の最後まで再生したら終了
        break

    # YOLO11でトラッキング（フレーム間でトラックを保持）
    result = model.track(frame, persist=True, tracker="custom_track.yaml")[0]

    # バウンディングボックスとトラックIDを取得（personのみ対象）
    current_ids = set()
    if result.boxes and result.boxes.is_track:

        # バウンディングボックスの長辺短辺の長さ
        boxes_wh = result.boxes.xywh.cpu()
        # バウンディングボックスの頂点座標
        boxes_xyxy = result.boxes.xyxy.cpu()
        # トラッキングID
        track_ids = result.boxes.id.int().cpu().tolist()
        # 分類クラス番号
        classes = result.boxes.cls.int().cpu().tolist() if result.boxes.cls is not None else [None] * len(track_ids)
        # 分類クラス一覧表
        names = result.names

        # マスク情報を取り出せるか試す
        masks = None
        try:
            if getattr(result, "masks", None) is not None:
                masks_data = getattr(result.masks, "data", result.masks)
                try:
                    masks = masks_data.cpu().numpy()
                except Exception:
                    masks = np.asarray(masks_data)
        except Exception:
            masks = None

        # バウンディングボックスを描画（ラベルなし）
        for idx, (box, box_xyxy, track_id, cls) in enumerate(zip(boxes_wh, boxes_xyxy, track_ids, classes)):
            obj_name = names.get(int(cls), "unknown") if cls is not None else "unknown"

            # 人以外のオブジェクトだった場合は描画しない
            # if obj_name != "person":
            #     continue
            # 今映っている人物一覧に登録
            current_ids.add(track_id)

            # 描画用の色を取得
            color = id_to_color(track_id)


            x1, y1, x2, y2 = box_xyxy
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            position_x = (x1 + x2) / 2.0
            position_y = (y1 + y2) / 2.0
            # 初期座標を登録
            if track_id not in all_person_ids:
                all_person_ids.add(track_id)
                first_positions[track_id] = (position_x, position_y)
                # 最初に検出されたフレームのカラー画像を保存
                try:
                    first_frames[track_id] = frame.copy()
                except Exception:
                    first_frames[track_id] = None
                # 最初に検出されたフレームのマスクを保存する（可能なら）
                if masks is not None and idx < len(masks):
                    try:
                        mask = masks[idx]
                        # mask が浮動小数点で 0-1 の場合や bool の場合に対応
                        if mask.dtype != np.uint8:
                            mask_img = (mask > 0.5).astype(np.uint8) * 255
                        else:
                            mask_img = (mask.copy()).astype(np.uint8)
                        # mask_img がフレームと同じ空間解像度でなければリサイズ
                        if mask_img.shape != frame.shape[:2]:
                            mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        first_masks[track_id] = mask_img
                    except Exception:
                        # マスク抽出に失敗したら無視
                        pass
                else:
                    # モデルからマスクが出力されない場合のフォールバック:
                    # 最初に検出されたバウンディングボックスを塗りつぶした単純マスクを作る
                    try:
                        mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                        # 画像外の座標をクリップ
                        x1i = max(0, min(frame.shape[1] - 1, x1i))
                        x2i = max(0, min(frame.shape[1] - 1, x2i))
                        y1i = max(0, min(frame.shape[0] - 1, y1i))
                        y2i = max(0, min(frame.shape[0] - 1, y2i))
                        cv2.rectangle(mask_img, (x1i, y1i), (x2i, y2i), 255, thickness=-1)
                        first_masks[track_id] = mask_img
                    except Exception:
                        pass
            # 最終座標を逐次登録
            last_positions[track_id] = (position_x, position_y)

            ### トラッキング処理 ###
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

    # ゾーン矩形を描画
    if zone_rect is not None:
        zx1, zy1, zx2, zy2 = zone_rect
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

    
    overlay_lines = []
    y_offset = 20
    for tid in sorted(all_person_ids):
        color = id_to_color(tid)
        
        # トラッキング線を描画
        track = track_history.get(tid, [])
        if len(track) >= 2:
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=10)
        
        # 画面左上に現在のID一覧をリスト表示
        y_offset += 25
        cv2.putText(
            frame,
            f"ID {tid}: person",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    last_frame = frame.copy()
    # 描画済みフレームを表示（GUI環境のみ）
    if not HEADLESS:
        cv2.imshow("YOLO11 Tracking", frame)

    # qキー:終了, sキー:一時停止/再開
    if not HEADLESS:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            paused = not paused
            while paused and not exit_requested:
                cv2.imshow("YOLO11 Tracking", frame)
                key_pause = cv2.waitKey(100) & 0xFF
                if key_pause == ord("s"):
                    paused = False
                elif key_pause == ord("q"):
                    exit_requested = True
                    break
    else:
        # ヘッドレス環境では自動で処理を継続
        pass
    
if last_frame is not None:
    overlay_lines = []
    y_offset = 45
    go_inside_count = 0
    go_outside_count = 0
    for tid in sorted(all_person_ids):
        color = id_to_color(tid)
        
        # トラッキング線を描画
        track = track_history.get(tid, [])
        if len(track) >= 2:
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(last_frame, [points], isClosed=False, color=color, thickness=10)
        
        # 画面左上に現在のID一覧をリスト表示
        

        (first_x, first_y) = first_positions[tid]
        (last_x, last_y) = last_positions[tid]
        cv2.circle(last_frame, (int(first_x), int(first_y)), 20, color, thickness=3, lineType=cv2.LINE_AA)
        cv2.drawMarker(last_frame, (int(last_x), int(last_y)), color, markerType=cv2.MARKER_CROSS, markerSize=200, thickness=3, line_type=cv2.LINE_AA)
        movement_x = last_x - first_x

        go_inside_count += 1 if (not is_inside_zone((first_x, first_y))) and is_inside_zone((last_x, last_y)) else 0
        go_outside_count += 1 if is_inside_zone((first_x, first_y)) and (not is_inside_zone((last_x, last_y))) else 0

        # 入退室判定
        cv2.putText(
            last_frame,
            f"ID {tid}: person (move { 'right' if movement_x > 0 else 'left'}) {'inside' if is_inside_zone((first_x, first_y)) else 'outside'} then {'inside' if is_inside_zone((last_x, last_y)) else 'outside'}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        y_offset += 25

    cv2.putText(
        last_frame,
        f"Total entered: {go_inside_count}, Total exited: {go_outside_count}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    # 描画済みフレームを表示（GUI環境のみ）
    if not HEADLESS:
        cv2.imshow("YOLO11 Tracking", last_frame)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # 毎回の実行ごとに専用フォルダを作成して出力する
    os.makedirs('result', exist_ok=True)
    run_dir = f'result/{video_filename}-{timestamp}'
    os.makedirs(run_dir, exist_ok=True)

    # 最終フレームを書き出し（run_dir/final.jpg）
    final_path = os.path.join(run_dir, 'final.jpg')
    cv2.imwrite(final_path, last_frame)

    # 各 person の最初に検出されたフレームのマスクを保存（run_dir/id{tid}_mask.png）
    for tid, mask_img in first_masks.items():
        try:
            # mask_img は単一チャネル (0/255) を想定
            # マスクを元にカラーで切り抜いた画像（透過PNG）を作成して保存
            # まず元フレーム（最初に検出されたフレーム）を取得
            src_frame = first_frames.get(tid, None)
            if src_frame is None:
                src_frame = last_frame

            # mask_img が 0/255 であることを仮定
            mask_bin = (mask_img > 0).astype(np.uint8) * 255

            # マスク領域の矩形でトリミング（余分な空白を削る）
            ys, xs = np.where(mask_bin > 0)
            if ys.size > 0 and xs.size > 0:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                # 切り抜き
                color_crop = src_frame[y1 : y2 + 1, x1 : x2 + 1].copy()
                alpha_crop = mask_bin[y1 : y2 + 1, x1 : x2 + 1].copy()

                # BGRA に変換してアルファチャンネルを設定
                if color_crop.ndim == 3 and color_crop.shape[2] == 3:
                    b, g, r = cv2.split(color_crop)
                    bgra = cv2.merge([b, g, r, alpha_crop])
                else:
                    # グレースケールだったら3チャンネル化
                    color_rgb = cv2.cvtColor(color_crop, cv2.COLOR_GRAY2BGR)
                    b, g, r = cv2.split(color_rgb)
                    bgra = cv2.merge([b, g, r, alpha_crop])

                crop_path = os.path.join(run_dir, f"id{tid}_person.png")
                cv2.imwrite(crop_path, bgra)
            else:
                # マスクがまったく無い場合は、スキップ
                pass
        except Exception:
            pass

            
            
        
# ビデオキャプチャを解放しウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()