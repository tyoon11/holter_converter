---


# Holter ECG Raw to HDF5 변환 파이프라인

이 프로젝트는 Holter ECG raw 데이터를 분석 가능한 구조로 변환하여 `.h5` 형식으로 저장하는 파이프라인을 제공합니다.

---

## 📦 의존성 설치

다음 Python 패키지를 설치해야 합니다:

```bash
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

## 🧪 HDF5 구조 확인

`h5_test.ipynb`를 통해 HDF5 구조와 데이터 내용을 탐색하고 시각화할 수 있습니다.

---
