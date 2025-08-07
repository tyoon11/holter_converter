물론이죠, 태윤님!
지금까지의 설명을 모두 반영한 **최종 `README.md` 통합본**을 아래에 제공해드릴게요. 그대로 복사해서 붙여 넣으면 바로 사용 가능합니다.

---

````markdown
# Holter ECG Raw to HDF5 변환 파이프라인

이 프로젝트는 Holter ECG raw 데이터를 분석 가능한 구조로 변환하여 `.h5` 형식으로 저장하는 파이프라인을 제공합니다.

---

## 📦 의존성 설치

다음 Python 패키지를 설치해야 합니다:

```
pip install pandas numpy h5py matplotlib neurokit2 dtw wfdb ray
````

---

## 📁 파일 구성

| 파일명                      | 설명                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `fix_pid.py`             | `.hea`, `.json` 파일 내의 PID 및 레코드명을 파일명 기준으로 정정합니다. **변환 전 필수 사전 작업입니다.**               |
| `utils.py`               | HDF5 변환에 필요한 유틸리티 함수들을 정의한 모듈입니다. (신호 품질, fiducial 포인트 추출, 유효 레코드 자동 생성 등 포함)         |
| `create_h5_structure.py` | ECG 세그먼트 기반의 HDF5 저장 구조를 정의하고, 데이터를 저장하는 함수들을 포함합니다.                                  |
| `convert_to_h5.py`       | raw 데이터를 `.h5` 형식으로 변환합니다. `valid_records.csv`가 없을 경우 자동으로 생성되며, `ray`를 사용해 병렬 처리됩니다. |
| `h5_test.ipynb`          | 생성된 `.h5` 파일의 구조를 탐색하고 내용을 시각화하는 Jupyter Notebook입니다.                                 |

---

## 📌 사전 정리: PID 및 레코드명 정정

Holter 원본 데이터는 `.hea`, `.json` 내부에 기록된 `record_name` 및 `PID`가 파일명과 일치하지 않는 경우가 있습니다.
이로 인해 이후 파이프라인에서 레코드 간 일관성이 깨질 수 있으므로, **파일명을 기준으로 내부 정보를 정정하는 사전 작업이 필요합니다.**

### ✅ fix\_pid.py 사용 방법

1. raw 파일 전체를 **read/write 권한이 있는 별도 폴더**로 복사합니다.
2. 아래 명령어를 실행하여 `.hea`, `.json` 파일의 내부 값을 정정합니다:

```bash
python fix_pid.py
```

> `.hea` 파일: 모든 줄의 `record_name`을 파일명으로 덮어씀
> `.json` 파일: `"PatientInfo"`의 `"PID"`를 파일명의 PID로 갱신

---

## 🚀 실행 흐름

1. `fix_pid.py` 실행
   → `.hea`, `.json` 내부의 레코드명을 파일명 기준으로 정정

   ```bash
   python fix_pid.py
   ```

2. `convert_to_h5.py` 실행
   → `.h5` 파일로 변환 (내부적으로 `valid_records.csv`가 없으면 자동 생성)

   ```bash
   python convert_to_h5.py
   ```

   ⚠️ 병렬 처리에 사용할 CPU 개수는 `ray.init(num_cpus=...)`에서 조절하세요.

---

## ⚙️ 주요 설정 옵션

`convert_to_h5.py` 실행 시 다음과 같은 옵션을 통해 처리 속도와 기능을 유연하게 조절할 수 있습니다:

| 옵션 이름                  | 설명                                                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| `use_dummy_fiducial`   | `True`로 설정 시, **fiducial point 및 ECG feature 추출을 생략**하고 `NaN` 및 빈 리스트로 채웁니다. 파이프라인의 실행 속도를 높이고자 할 때 유용합니다. |
| `use_dummy_similarity` | `True`로 설정 시, **beat 간 유사도(correlation, DTW distance) 계산을 생략**합니다. 유사하게 `NaN`으로 대체됩니다.                     |

> 예시:
>
> ```python
> convert_folder_to_h5_ray(
>     input_dir="...",
>     output_dir="...",
>     use_dummy_fiducial=True,
>     use_dummy_similarity=True
> )
> ```

---

## ⛔️ 이미 변환된 파일 건너뛰기

`convert_to_h5.py` 실행 시, **output 디렉토리에 동일한 이름의 `.h5` 파일이 이미 존재하면 해당 레코드는 자동으로 건너뜁니다.**
이는 불필요한 중복 계산을 방지하며, 중단 후 재시작 시 효율적으로 이어서 처리할 수 있도록 설계되었습니다.

---

## 🔄 예시 실행 스크립트

```python
convert_folder_to_h5_ray(
    input_dir="/your/raw/data",
    output_dir="/your/output/h5",
    csv_path="output_h5_list.csv",
    sampling_rate=125,
    segment_sec=10,
    log_path="conversion_log.txt",
    valid_list_path="valid_records.csv",
    use_dummy_fiducial=True,
    use_dummy_similarity=True
)
```

---

## 🧪 HDF5 구조 확인

`h5_test.ipynb`를 통해 생성된 `.h5` 파일의 구조와 데이터를 탐색하고 시각화할 수 있습니다.


---

