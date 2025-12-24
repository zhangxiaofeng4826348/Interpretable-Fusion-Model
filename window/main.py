from PIL import Image, ImageTk
import cv2
import os
import csv
import sys
import tkinter as tk
from tkinter import messagebox


def get_resource_path(relative_path):
    """获取资源的绝对路径"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件
        base_dir = os.path.dirname(sys.executable)
    else:
        # 如果是脚本运行
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)



class UserSelectionWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasonic image discrimination")  # 第一个窗口名字改为“超声图像鉴别”
        self.root.geometry("800x400")  # 设置初始窗口大小

        # 用户身份选择
        self.user_type = tk.StringVar(value="Radiologist")

        # 标题
        title_label = tk.Label(root, text="Please select your identity", font=("Arial", 24))
        title_label.pack(pady=50)

        # 单选按钮
        user_frame = tk.Frame(root)
        user_frame.pack()

        tk.Radiobutton(user_frame, text="Senior radiologist", variable=self.user_type, value="Senior radiologist", font=("Arial", 18)).pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(user_frame, text="Junior radiologist", variable=self.user_type, value="Junior radiologist", font=("Arial", 18)).pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(user_frame, text="Professor", variable=self.user_type, value="Professor", font=("Arial", 18)).pack(side=tk.LEFT, padx=20)

        # 确认按钮
        confirm_button = tk.Button(root, text="Confirm", command=self.confirm_selection, font=("Arial", 18))
        confirm_button.pack(pady=50)

    def confirm_selection(self):
        user_type = self.user_type.get()
        self.root.destroy()  # 关闭当前窗口
        start_main_app(user_type)  # 启动主程序


class UltrasoundApp:
    def __init__(self, root, user_type):
        self.root = root
        self.user_type = user_type

        # 设置窗口标题为“超声图像鉴别（具体的用户身份）”
        self.root.title(f"Ultrasonic image discrimination（{user_type}）")

        # 设置窗口初始大小
        self.window_width = 1200  # 窗口高度
        self.window_height = int(self.window_width / 1.25)  # 窗口宽度
        self.root.geometry(f"{self.window_width}x{self.window_height}")

        # 图像序号描述
        self.image_info_label = tk.Label(root, text="", font=("Arial", 16))
        self.image_info_label.pack(pady=10)

        # 图像展示与交互
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # 问题与选项
        self.question_label = tk.Label(root, text="Please select the most likely case of occupying space in the ultrasound image", font=("Arial", 20))
        self.question_label.pack(pady=10)

        self.option_var = tk.StringVar(value="")  # 用于存储用户选择的选项
        self.create_options()

        self.submit_button = tk.Button(root, text="Confirm", command=self.submit_answer, font=("Arial", 16))
        self.submit_button.pack(pady=20)

        # 自动加载图像
        self.image_folder = get_resource_path("image")  # 修改为动态路径
        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.current_image_index = 0

        # 存储用户答案
        self.user_answers = []

        # 加载第一张图片
        self.load_next_image()

    def create_options(self):
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=10)

        options = ["A: Thrombus", "B: Nonthrombus", "C: Unclear"]
        for option in options:
            tk.Radiobutton(options_frame, text=option, variable=self.option_var, value=option, font=("Arial", 16)).pack(anchor=tk.W)

    def load_next_image(self):
        if self.current_image_index < len(self.image_files):
            # 更新图像序号描述
            self.image_info_label.config(
                text=f"Current ultrasonic image：{self.current_image_index + 1}/{len(self.image_files)}  Remain：{len(self.image_files) - self.current_image_index - 1}"
            )

            # 加载图像
            image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(self.current_image)

            # 旋转图像（例如旋转90度）
            self.current_image = self.current_image.rotate(270, expand=True)

            # 固定图像大小
            image_width = 850  # 固定宽度
            image_height = 600  # 固定高度
            resized_image = self.current_image.resize((image_width, image_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)

            # 更新图像显示
            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.current_image_index += 1
        else:
            # 所有图像加载完毕，导出用户答案
            self.export_answers()
            messagebox.showinfo("Finish", "All images are loaded!")
            self.root.quit()

    def submit_answer(self):
        answer = self.option_var.get()
        if answer:
            # 记录用户答案
            self.user_answers.append((self.current_image_index, answer))

            # 清空选择的记忆
            self.option_var.set("")

            # 直接加载下一张图片，不弹出提交成功提示
            self.load_next_image()
        else:
            messagebox.showwarning("Commit failure", "Please select an option first.")

    def export_answers(self):
        # 获取可执行文件所在的目录
        if getattr(sys, 'frozen', False):
            # 如果是打包后的可执行文件
            base_dir = os.path.dirname(sys.executable)
        else:
            # 如果是脚本运行
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建 CSV 文件路径
        csv_file = os.path.join(base_dir, "user_answers.csv")

        # 导出用户身份和所有选择答案为 CSV 文件
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["User identity", "Picture serial number", "User answer"])
            for index, answer in self.user_answers:
                writer.writerow([self.user_type, index + 1, answer])

        # 提示导出成功
        messagebox.showinfo("Export successfully", f"Exported to {csv_file}")


def start_main_app(user_type):
    root = tk.Tk()
    app = UltrasoundApp(root, user_type)
    root.mainloop()


if __name__ == "__main__":
    # 启动用户选择窗口
    root = tk.Tk()
    user_selection_window = UserSelectionWindow(root)
    root.mainloop()