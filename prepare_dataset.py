import os
import shutil
import pandas as pd
import chardet

# === Paths ===
training_folder = "RFMiD2_Dataset/Training_set"
label_file = os.path.join(training_folder, "Training_labels.csv")
output_dir = "dataset"

# === Output folders ===
heart_disease_dir = os.path.join(output_dir, "heart_disease")
no_heart_disease_dir = os.path.join(output_dir, "no_heart_disease")
os.makedirs(heart_disease_dir, exist_ok=True)
os.makedirs(no_heart_disease_dir, exist_ok=True)

# === Check paths ===
if not os.path.exists(training_folder):
    print(f"❌ Training folder not found: {training_folder}")
    exit()

if not os.path.exists(label_file):
    print(f"❌ Label file not found: {label_file}")
    exit()

# === Detect file encoding ===
with open(label_file, 'rb') as f:
    result = chardet.detect(f.read())
    detected_encoding = result['encoding']
    print(f"🔍 Detected encoding: {detected_encoding}")

# === Read label file with encoding fallback ===
try:
    df = pd.read_csv(label_file, encoding=detected_encoding)
    print("✅ CSV loaded successfully!")
except UnicodeDecodeError:
    print("❌ Unicode error. Retrying with 'latin1'.")
    df = pd.read_csv(label_file, encoding='latin1')
except Exception as e:
    print(f"❌ Failed to load CSV: {e}")
    exit()

# === Check necessary columns ===
if 'ID' not in df.columns or 'WNL' not in df.columns:
    print("❌ Required columns ('ID', 'WNL') are missing in the CSV file.")
    exit()

# === Convert ID column to integers ===
df['ID'] = df['ID'].astype(float).astype(int)

# === Process images ===
missing_files = 0
missing_list = []
copied_heart = 0
copied_no_heart = 0

for _, row in df.iterrows():
    image_name = f"{int(row['ID'])}.jpg"
    src = os.path.join(training_folder, image_name)

    if not os.path.exists(src):
        print(f"❌ File not found: {src}")
        missing_list.append(image_name)
        missing_files += 1
        continue

    # WNL = 1 means normal → no heart disease
    if row['WNL'] == 1:
        dst = os.path.join(no_heart_disease_dir, image_name)
        copied_no_heart += 1
    else:
        dst = os.path.join(heart_disease_dir, image_name)
        copied_heart += 1

    shutil.copy(src, dst)

# === Save missing files to log ===
if missing_list:
    with open("missing_images.txt", "w") as f:
        f.write("\n".join(missing_list))

# === Summary ===
print("\n📦 Dataset preparation complete!")
print(f"✅ Images with no heart disease: {copied_no_heart}")
print(f"✅ Images with heart disease: {copied_heart}")
print(f"❌ Total missing files: {missing_files}")
print(f"📝 Missing file names saved to: missing_images.txt")


import matplotlib.pyplot as plt

# === Pie chart of class distribution ===
labels = ['No Heart Disease (WNL=1)', 'Heart Disease (WNL≠1)']
sizes = [copied_no_heart, copied_heart]
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0.05)  # Slightly separate the slices

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Heart Disease vs No Heart Disease (WNL)')
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.tight_layout()
plt.savefig("class_distribution_pie_chart.png")
plt.show()
