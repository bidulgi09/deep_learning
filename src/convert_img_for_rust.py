import os
import cv2
import numpy as np
import gc
from pathlib import Path

# 1. 경로 및 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  
src_dir = project_root / "PokemonData"

IMG_SIZE = 128  
SPLIT_RATIO = 0.8  

# 🛑 [공간 확보 레벨 1] 용량을 대거 차지하는 불필요한 압축 파일들 강제 제거
zip_files_to_clean = [
    project_root / "deep_learning.zip",
    project_root / "pokemonclassification.zip"
]

print("=== DISK SPACE CLEANUP ===")
for zip_path in zip_files_to_clean:
    if zip_path.exists():
        print(f"Removing heavy file to free space: {zip_path.name}")
        try:
            zip_path.unlink() # 파일 영구 삭제
        except Exception as e:
            print(f"Failed to remove {zip_path.name}: {e}")

train_images, train_labels = [], []
val_images, val_labels = [], []

# 2. 클래스(포켓몬 이름) 정렬 및 인덱스 맵핑
class_names = sorted([f.name for f in src_dir.iterdir() if f.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

print("\n=== START CONVERTING ===")
print(f"Dataset Path: {src_dir.resolve()}")
print(f"Total Pokemon Classes: {len(class_names)}")

# 3. 이미지 루프 및 전처리
for class_name in class_names:
    class_folder = src_dir / class_name
    label_idx = class_to_idx[class_name]
    
    img_paths = [p for p in class_folder.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    np.random.shuffle(img_paths)
    
    split_idx = int(len(img_paths) * SPLIT_RATIO)
    
    for i, img_path in enumerate(img_paths):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            img_data = img.astype(np.float32) / 255.0  
            img_data = np.transpose(img_data, (2, 0, 1)) 
            
            if i < split_idx:
                train_images.append(img_data)
                train_labels.append(label_idx)
            else:
                val_images.append(img_data)
                val_labels.append(label_idx)
        except Exception:
            pass

print("Processing and allocating arrays...")

# 4. 넘파이 배열 변환 
X_train = np.array(train_images, dtype=np.float32)
y_train = np.array(train_labels, dtype=np.int32)
X_val = np.array(val_images, dtype=np.float32)
y_val = np.array(val_labels, dtype=np.int32)

# 🛑 [공간 확보 레벨 2] 힙 메모리에 남아있는 리스트 찌꺼기 가비지 컬렉션으로 비우기
del train_images, train_labels, val_images, val_labels
gc.collect()

# 전체 데이터 셔플
idx_train = np.arange(X_train.shape[0])
np.random.shuffle(idx_train)
X_train, y_train = X_train[idx_train], y_train[idx_train]

idx_val = np.arange(X_val.shape[0])
np.random.shuffle(idx_val)
X_val, y_val = X_val[idx_val], y_val[idx_val]

# 5. 분할된 데이터셋 저장
output_path = project_root / "pokemon_dataset_split.npz"
print(f"Saving compressed file to {output_path.name}...")

np.savez_compressed(
    str(output_path), 
    x_train=X_train, y_train=y_train,
    x_val=X_val, y_val=y_val
)

print("\n=========================================")
print("SUCCESS: Dataset conversion completed!")
print(f"Saved Path: {output_path.resolve()}")
print(f"Train Shape: {X_train.shape} | Val Shape: {X_val.shape}")
print("=========================================")
