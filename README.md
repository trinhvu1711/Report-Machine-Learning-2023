# Báo cáo Máy Học

### Nhóm 18: Trịnh Long Vũ, Trần Bùi Tuấn Ngọc

# Dự đoán đội chiến thắng trong trận đấu giải Ngoại Hạng Anh - Premier League

## 1. Giới thiệu

Bóng đá là bộ môn thể thao phổ biến nhất trên thế giới. Một số quốc gia có nhiều đội bóng thi đấu ở giải đấu vô địch khu vực và cấp quốc gia. Trong số các giải vô địch quốc gia trên thế giới thì đề tài này chủ yếu tập trung vào giải Ngoại Hạng Anh (Premier League), nơi giải đấu có nhiều người xem nhất trên thế giới.

### 1.1. Mục tiêu

- Xây dựng được mô hình dự đoán một trận đấu bóng đá và so sánh độ chính xác giữa các phương pháp phân lớp (đạt độ chính xác hơn 50% và duy trì được 60%).
- Giải thích được các yếu tố ảnh hưởng đến kết quả của các trận đấu diễn ra.

### 1.2. Giới hạn

- Dữ liệu (dataset) được lấy từ trang [fbref](https://fbref.com/en/comps/9/Premier-League-Stats)
- Thuật toán được sử dụng để xây dựng mô hình gồm có: Decision Tree, RandomForest, SVM, hồi quy Logistic, kNN và Neural Network

## 2. Chuẩn bị dữ liệu

### 2.1. Thu thập dữ liệu

Dữ liệu được tổng hợp ở bảng: "Regular season" [ở đây](https://fbref.com/en/comps/9/Premier-League-Stats) và chi tiết dữ liệu cho từng đội ở bảng "Shotting"

Đầu tiên, ta khai báo danh sách các năm cần thu thập dữ liệu (từ năm 2022 đến năm 2020) trong biến years.

Tiếp theo, ta khai báo một danh sách rỗng all_matches để lưu trữ dữ liệu thu thập được.

Ta khai báo biến standings_url để lưu trữ đường dẫn tới trang web chứa thông tin bảng xếp hạng và thống kê.

```python
years = list(range(2022, 2020, -1))
all_matches = []

standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
```

Sử dụng vòng lặp for để duyệt qua từng năm trong danh sách years.

Trong vòng lặp, ta sử dụng thư viện requests và BeautifulSoup để lấy dữ liệu từ standings_url và tạo đối tượng soup để xử lý dữ liệu HTML.
Ghép các bảng dữ liệu trận đấu và thống kê cú sút thành một bảng dữ liệu hoàn chỉnh của mỗi đội bóng.Lọc ra chỉ các trận đấu trong giải Ngoại Hạng Anh bằng cách kiểm tra cột "Comp" trong bảng dữ liệu.

```python
import time
for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]

        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(1)
```

Dữ liệu ta tổng hợp được có các đặc trưng:

- date: Ngày diễn ra trận đấu
- time: Thời gian diễn ra trận đấu
- comp: Giải đấu
- round: Vòng đấu
- day: Thứ trong tuần
- venue: Địa điểm tổ chức trận đấu
- result: Kết quả trận đấu
- gf: Số bàn thắng của đội chủ nhà
- ga: Số bàn thua của đội chủ nhà
- opponent: Đội đối thủ
- xg: Xếp hạng xG (Expected Goals) của đội chủ nhà
- xga: Xếp hạng xG (Expected Goals) của đội đối thủ
- poss: Tỷ lệ kiểm soát bóng của đội chủ nhà
- attendance: Số lượng khán giả có mặt
- captain: Đội trưởng của đội chủ nhà
- formation: Hệ thống chiến thuật của đội chủ nhà
- referee: Trọng tài
- match report: Báo cáo trận đấu
- notes: Ghi chú
- sh: Số cú sút của đội chủ nhà
- sot: Số cú sút trúng đích của đội chủ nhà
- dist: Tổng quãng đường chạy của đội chủ nhà
- fk: Số lượt sút phạt của đội chủ nhà
- pk: Số lượt sút penalty của đội chủ nhà
- pkatt: Số lượt sút penalty thực hiện của đội chủ nhà
- season: Mùa giải
- team: Tên đội chủ nhà

Các cột dữ liệu trên cung cấp thông tin về kết quả, số liệu thống kê và các yếu tố liên quan đến trận đấu và đội bóng chủ nhà.

Các cột như "gf", "ga", "xg", "xga", "poss", "sh", "sot", "dist", "fk", "pk", "pkatt" là các chỉ số thể hiện khả năng tấn công và phòng thủ của đội bóng chủ nhà trong trận đấu.

Thông qua việc phân tích và xử lý dữ liệu này, ta có thể trích chọn đặc trưng (Feature Selection) để dự đoán kết quả và đội chiến thắng trong các trận đấu giải Ngoại Hạng Anh.

### 2.2. Sơ chế dữ liệu (Data Wrangling)

Xóa các đặc trưng không cần thiết

```python
del matches["comp"]
del matches["notes"]
```

Chuyển đổi dữ liệu đúng định dạng

```python
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
```

### 2.3. Trực quan hóa dữ liệu (Data Visualisation)

<img src="Data Visualisation\1.png" alt="Figure 1">
 
<em>Figure 1. Trực quan hóa phân phối của số bàn thắng của đội chủ nhà</em>

<img src="Data Visualisation\2.png" alt="Figure 2">

<em>Figure 2. Biểu đồ tương quan giữa tỷ lệ kiểm soát bóng và số cú sút của đội chủ nhà</em>

<img src="Data Visualisation\3.png" alt="Figure 3">

<em>Figure 3.Biểu đồ boxplot cho xếp hạng xG của đội chủ nhà theo giải đấu</em>

<img src="Data Visualisation\newplot.png" alt="Figure 4">

<em>Figure 4. Biểu đồ tương quan giữa số lượng khán giả có mặt và số bàn thắng đội nhà</em>

<img src="Data Visualisation\4.png" alt="Figure 5">

<em>Figure 5. Biểu đồ heatmap để hiển thị ma trận tương quan giữa các biến</em>

### 2.4. Trích chọn đặc trưng (Feature Selection)

Thực hiện trích chọn đặc trưng, sử dụng SelectKBest để chọn K đặc trưng quan trọng nhất

<img src="Data Visualisation\5.png" alt="Figure 5">

<em>Figure 6. Biểu đồ các đặc trưng được chọn</em>

## 3. Chọn mô hình và huấn luyện

Đồ án sẽ xây dựng các mô hình giúp dự đoán kết quả trận đấu.

Lấy dữ liệu 'matches 2019-2023.xls'
Phân chia dữ liệu thành 2 tập: Training Set ( date<'2022-01-01') và Test Set (date >'2022-01-01').

Khởi tạo mô hình, nhận thuật toán phân lớp thông qua tham số clf (classifier) của hàm.

Đánh giá mô hình bằng Kiểm chứng chéo (Cross Validation) trên Training Set

Đánh giá 4 hệ số: Accuracy (Độ chính xác tổng quát), Precision (Độ chính xác), Recall (Độ nhạy), F1.

Vẽ biểu đồ hộp cho 4 hệ số đánh giá từ kết quả kiểm chứng chéo Training Set.

Vẽ Confusion Matrix cho kết quả dự đoán của mô hình trên Test Set.

Các thuật toán lựa chọn và sử dụng từ thư viện scikit-learn. Tất cả mô hình được tối ưu hóa bằng cách sử dụng grid search

- Decision Tree
- SVM
- K-Nearest Neighbour
- Hồi quy Logistic
- RandomForest
- Neural Network

### 3.1. Đánh giá nhanh độ chính xác của mô hình

Ta kiểm tra nhanh độ chính xác của mô hình bằng cách sử dụng thuật toán RandomForest

```
accuracy score:  0.600739371534196
precision score:  0.48736462093862815
recall score:  0.3176470588235294
f1 score:  0.3846153846153846

| predicted |  0  |  1  |
| --------- | :-: | :-: |
| actual    |     |     |
| 0         | 515 | 142 |
| 1         | 290 | 135 |
```

### 3.2. Cải thiện độ chính xác cho mô hình
