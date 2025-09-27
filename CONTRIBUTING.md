# Contribution Guidelines

Cảm ơn bạn đã quan tâm đến việc đóng góp cho dự án này. Để đảm bảo tính nhất quán và chất lượng mã nguồn, vui lòng tuân thủ các hướng dẫn sau.

## 1. Cấu trúc mã và chuẩn lập trình

- Tuân thủ cấu trúc thư mục hiện có: `core/`, `api/`, `scripts/`, `tests/`.
- Mỗi module mới nên có:
  - Chú thích `docstring` rõ ràng cho hàm/lớp
  - Kiểu dữ liệu (type annotation) khi có thể
  - Ghi log cho các phần dễ lỗi hoặc ngoại lệ
- Không hardcode đường dẫn; hãy sử dụng cấu hình.

## 2. Quy tắc viết commit

Sử dụng định dạng `conventional commit`:

<type>(<scope>): <short summary>

**type** bao gồm:
- `feat`: thêm tính năng mới
- `fix`: sửa lỗi
- `refactor`: cải tổ mã không thay đổi logic
- `test`: thêm/sửa test
- `docs`: tài liệu
- `chore`: thay đổi phụ trợ (CI, dependency, v.v.)
- `style`: định dạng mã (không ảnh hưởng logic)

**scope** chỉ phạm vi thay đổi, ví dụ: `chunking`, `embed`, `api`, `test`, `script`, v.v.

**Ví dụ:**
- `feat(embed): implement FLAVA embedding wrapper`
- `fix(api): handle missing file uploads in POST route`

## 3. Kiểm thử (Testing)

- Mỗi tính năng mới cần có test tương ứng trong thư mục `tests/`.
- Sử dụng `pytest` để chạy test.
- Test phải độc lập và có thể lặp lại, không phụ thuộc vào mạng hoặc dịch vụ ngoài.

## 4. Tài liệu

- Cập nhật `README.md` nếu thay đổi ảnh hưởng đến cách sử dụng.
- Viết chú thích trong mã rõ ràng với các đoạn logic phức tạp.
- Nếu có thể, cung cấp ví dụ về input/output.

## 5. Môi trường và phụ thuộc

- Tất cả thư viện sử dụng cần được khai báo trong `requirements.txt`.
- Không commit các file lớn (.mp4, .json) hoặc thư mục môi trường ảo (venv, .env).

## 6. Pull Request

- Gửi PR lên nhánh `main`.
- Mô tả rõ ràng thay đổi bạn thực hiện và lý do.
- Nếu có liên kết tới Issue, hãy dùng định dạng `Closes #<số>`.

## 7. Định dạng mã (Formatting)

- Chạy `black .` để chuẩn hóa định dạng.
- Kiểm tra kiểu dữ liệu bằng `mypy`.
- CI sẽ từ chối PR nếu test hoặc định dạng sai.

## 8. Trao đổi và liên hệ

- Mọi yêu cầu tính năng, báo lỗi nên thông qua GitHub Issue.
- Giữ trao đổi kỹ thuật, rõ ràng, và chuyên nghiệp.

Chúng tôi hoan nghênh mọi đóng góp nghiêm túc và có tổ chức. Trân trọng cảm ơn.
