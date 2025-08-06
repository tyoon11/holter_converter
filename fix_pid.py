import os
import json
import stat
from tqdm import tqdm


def fix_hea_and_json_by_filename(input_dir):
    """
    ✅ .hea 파일: 모든 줄의 record_name을 파일명 그대로로 덮어쓰기
    ✅ .json 파일: PatientInfo → PID 값을 파일명에서 추출한 PID로 덮어쓰기
    """
    files = os.listdir(input_dir)
    base_names = sorted(set(os.path.splitext(f)[0] for f in files))

    for base_name in tqdm(base_names, desc="🛠 PID 및 record_name 정정 중", ncols=100):
        record_name = base_name  # ex: DVD20160909_33_2411480
        pid = base_name.split("_")[-1]  # ex: 2411480

        hea_path = os.path.join(input_dir, base_name + ".hea")
        json_path = os.path.join(input_dir, base_name + ".json")

        # ───────────────
        # ✅ 1. .hea 수정
        # ───────────────
        if os.path.exists(hea_path):
            try:
                with open(hea_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = record_name  # ✅ 파일명 기준
                        new_lines.append(" ".join(parts) + "\n")
                    else:
                        new_lines.append(line)

                try:
                    with open(hea_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                except PermissionError:
                    os.chmod(hea_path, stat.S_IWUSR | stat.S_IRUSR)
                    with open(hea_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)

            except Exception:
                continue  # 예외 무시

        # ────────────────
        # ✅ 2. .json 수정
        # ────────────────
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "Holter Report" in data:
                    data["Holter Report"].setdefault("PatientInfo", {})
                    data["Holter Report"]["PatientInfo"]["PID"] = pid

                    try:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    except PermissionError:
                        os.chmod(json_path, stat.S_IWUSR | stat.S_IRUSR)
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                continue


if __name__ == "__main__":
    fix_hea_and_json_by_filename("/home/coder/workspace/data/tykim/nas1_Holter_PSVT")
