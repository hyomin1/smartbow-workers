from ultralytics import YOLO

model = YOLO("person_best.pt")

print("변환 시작... 5~10분 걸립니다")

model.export(
    format="engine",
    opset=18,
    dynamic=False,
    simplify=True,
    half=False,
    device=0,
    workspace=4,
)

print("완료!")
