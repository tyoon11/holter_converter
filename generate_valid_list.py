# This script generates a CSV file containing valid records from a specified directory.

import os
import pandas as pd


def has_all_required_files(base_path):
    return all(
        os.path.exists(base_path + ext) for ext in [".hea", ".SIG", ".ANN", ".json"]
    )


def generate_valid_records(input_dir, output_csv="valid_records.csv"):
    hea_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".hea")]
    record_names = [os.path.splitext(f)[0] for f in hea_files]

    valid_records = []
    for name in record_names:
        base_path = os.path.join(input_dir, name)
        if has_all_required_files(base_path):
            valid_records.append(name)

    df = pd.DataFrame({"record_name": valid_records})
    df.to_csv(output_csv, index=False)
    print(f"✅ 유효한 레코드 {len(valid_records)}개 저장됨: {output_csv}")


# 실행 예시
if __name__ == "__main__":
    generate_valid_records(
        input_dir="/home/coder/workspace/nas1_Holter_PSVT_250514",
        output_csv="/home/coder/workspace/data/tykim/valid_records.csv",
    )
