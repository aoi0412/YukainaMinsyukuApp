from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# YOLO11モデルを読み込む
model = YOLO("yolo11n.pt")

# 動画ファイルを開く
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

# トラッキング履歴を保存する辞書
track_history = defaultdict(lambda: [])
paused = False
exit_requested = False

# 入退室判定用のゾーン矩形（左上x1,y1, 右下x2,y2）
zone_rect = (100, 100, 400, 400)

# 各IDの最後の中心座標と入退室ラベル
last_positions = {}
event_labels = {}
active_ids_prev = set()


def is_inside_zone(pos):
    x1, y1, x2, y2 = zone_rect
    x, y = pos
    return x1 <= x <= x2 and y1 <= y <= y2


# 動画フレームをループ処理
while cap.isOpened():
    # 動画から1フレーム読み込む（ポーズ中は読み込まない）
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
        boxes_wh = result.boxes.xywh.cpu()
        boxes_xyxy = result.boxes.xyxy.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist() if result.boxes.cls is not None else [None] * len(track_ids)
        names = result.names

        # バウンディングボックスを描画（ラベルなし）
        for box_xyxy, track_id, cls in zip(boxes_xyxy, track_ids, classes):
            obj_name = names.get(int(cls), "unknown") if cls is not None else "unknown"
            if obj_name != "person":
                continue
            current_ids.add(track_id)
            x1, y1, x2, y2 = box_xyxy
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 255), thickness=2)
            last_positions[track_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # トラック軌跡を描画
        for box, track_id, cls in zip(boxes_wh, track_ids, classes):
            obj_name = names.get(int(cls), "unknown") if cls is not None else "unknown"
            if obj_name != "person":
                continue
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, yの中心座標
            if len(track) > 30:  # 30フレーム分の軌跡を保持
                track.pop(0)

            # トラッキング線を描画
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # フェードアウトしたIDに入退室ラベルを付与
        disappeared_ids = active_ids_prev - current_ids
        for tid in disappeared_ids:
            if tid in last_positions and tid not in event_labels:
                event_labels[tid] = "入室" if is_inside_zone(last_positions[tid]) else "退室"
        active_ids_prev = current_ids

    # ゾーン矩形を描画（処理前の目印）
    zx1, zy1, zx2, zy2 = zone_rect
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

    # 画面左上に現在のIDとフェードアウト済みの入退室ラベルをリスト表示
    overlay_lines = []
    for tid in sorted(current_ids):
        overlay_lines.append(f"ID {tid}: person")
    for tid in sorted(event_labels):
        overlay_lines.append(f"ID {tid}: {event_labels[tid]}")

    for idx, line in enumerate(overlay_lines):
        y_offset = 20 + idx * 25
        cv2.putText(
            frame,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 描画済みフレームを表示
    cv2.imshow("YOLO11 Tracking", frame)

    # qキー:終了, sキー:一時停止/再開
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

# ビデオキャプチャを解放しウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
