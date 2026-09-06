## Phân loại PR:
- [x] Feature
- [ ] Bugs
- [ ] Hotfix

# Đối chiếu tính năng đã làm theo yêu cầu:
- [x] #1: Xây dựng bộ lọc đọc ảnh tuần tự bằng OpenCV Generator chống tràn RAM.
- [x] #2: Hiện thực hóa thuật toán so khớp không-thời gian Tracker bảo toàn điểm số STIoU.

# Cách thực hiện để xử lý mỗi yêu cầu:
- #1: Viết lại phân hệ Generator giải phóng ma trận pixel ngay sau mỗi frame xử lý trong `video_reader.py`.
- #2: Thêm lớp đối tượng `SimpleIoUTracker` liên kết ID vật thể liên tục qua các khung hình bị che khuất tại `object_tracker.py`.

# Phạm vi ảnh hưởng:
- Thay đổi thuật toán đọc làm tăng nhẹ thời gian suy luận trên CPU nhưng ép chặt RAM cố định ở mức an toàn ~1.5GB, giải quyết triệt để lỗi Terminated hệ thống.

# Liên kết đến các issues liên quan:

- Gắn link issue gốc
- Gắn link issue hoặc PR mà việc thay đổi gây ảnh hưởng đến.

