import tkinter as tk
import random
import time
import numpy as np
import datetime

# 设置参数
start_up_value = 10  # 实验准备时间
task_time_value = 12  # 测试时间
rest_time_value = 15  # 休息时间
trail_value = 20    # 测试次数

# 数字记忆广度参数
forward_span = 5  # 前向记忆部分的序列长度
backward_span = 5  # 逆向记忆部分的序列长度

# list记录是每次激活态结束和休息开始的时候，因此mark-task_time_value是激活态开始

class StroopExperiment(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Stroop")
        self.geometry("500x500")
        self.create_stimulus()
        self.colors = ['red', 'green', 'blue', 'yellow']
        self.color_names = ['红', '绿', '蓝', '黄']
        self.trail_max = trail_value+1
        self.trail_num = trail_value
        self.task_time = task_time_value
        self.listT=[]

    def sav_list(self):
        with open('stroop_mark.txt', 'a', encoding='utf-8') as f:
            f.write('\n\n####################################################################################\n')
            f.write(str(datetime.datetime.now()))
            f.write('########################################################\n')
            for j in self.listT:
                f.write(str(j) + str('    '))
            f.write('\n####################################################################################\n\n\n\n\n')
            f.close()

        # 第一步进入欢迎界面，介绍试验流程
    def create_stimulus(self, start_up_time = start_up_value):
        self.prompt_label1 = tk.Label(self, text="欢迎参加 Stroop 任务！")
        self.prompt_label1.pack(pady=10)

        self.prompt_label2 = tk.Label(self, text="点击开始后将开始后将进入{}s倒计时，倒计时结束的瞬间请开始记录试验数据".format(start_up_time),font=("Arial", 10))
        self.prompt_label2.pack(pady=10)

        self.word_label = tk.Label(self, text="", font=("Helvetica", 24))
        self.word_label.pack()

        self.entry = tk.Entry(self, width=20, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        self.check_button = tk.Button(self, text="检查答案", command=self.check_answer)
        self.check_button.pack()

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.next_button = tk.Button(self, text="开始", command=self.start_up)
        self.next_button.pack(side=tk.BOTTOM, pady=10)

    # 试验开始倒计时
    def start_up(self,remaining = start_up_value):
        if remaining > 0:
            self.prompt_label1.config(text="实验将在{}秒后开始".format(remaining))
            self.next_button.config(text="",command=lambda: None)
            self.after(1000, self.start_up, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.start = self.sti
            self.stroop_task()

    def check_answer(self):
        self.user_input = self.entry.get().lower()
        if self.user_input == self.target_color:
            self.result_label.config(text="回答正确！", fg="green")
        else:
            self.result_label.config(text="回答错误！正确答案是" + self.target_color, fg="red")

    def rest_time(self,remaining = rest_time_value):

        if remaining > 0:
            self.prompt_label1.config(text="休息时间")
            self.prompt_label2.config(text="实验将在{}秒后开始".format(remaining))
            self.word_label.config(text="")
            self.result_label.config(text="")
            self.next_button.config(text="", command=lambda: None)
            self.after(1000, self.rest_time, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.trail_num -= 1
            self.stroop_task()

    def stroop_task(self):
        self.end = time.perf_counter()
        self.target_color = random.choice(self.colors)
        self.target_name = random.choice(self.color_names)
        self.prompt_label1.config(text="第{}轮测试".format(self.trail_max-self.trail_num))
        self.prompt_label2.config(text="请忽略下面的单词，注意看单词的颜色：")
        self.word_label.config(text=self.target_name, fg=self.target_color)
        self.result_label.config(text="")
        if ((self.end - self.sti) < self.task_time) & (self.trail_num > 0):
            self.next_button.config(text="下一个", command=self.stroop_task)
        elif ((self.end - self.sti) >= self.task_time) & (self.trail_num > 0):
            self.mark = self.end-self.start
            self.listT.append(self.mark)
            self.rest_time()
        elif self.trail_num <= 0:
            self.sav_list()
            self.destroy()


class MentalCalculationExperiment(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("MentalCalculation")
        self.geometry("500x500")
        self.create_stimulus()
        self.ops = ['+', '-']
        self.trail_max = trail_value + 1
        self.trail_num = trail_value
        self.task_time = task_time_value
        self.cmds = {'+': self.add, '-': self.sub}
        self.listT = []


    def sav_list(self):
        with open('MentalCalculation_mark.txt', 'a', encoding='utf-8') as f:
            f.write('\n\n####################################################################################\n')
            f.write(str(datetime.datetime.now()))
            f.write('########################################################\n')
            for j in self.listT:
                f.write(str(j) + str('    '))
            f.write('\n####################################################################################\n\n\n\n\n')
            f.close()

    def add(self, x, y):
        return x + y

    def sub(self, x, y):
        return x - y

    def high_level(self):
        num_range = 3
        num_digit_list = np.zeros([3, num_range])
        op1 = random.choice(self.ops)
        op2 = random.choice(self.ops)
        if op1 == '+':
            for i in range(num_range):
                num_digit_list[0, i] = np.random.randint(2, 9)
            num_digit_list[1, 0] = np.random.randint(10 - int(num_digit_list[0, 0]), 9)
            num_digit_list[1, 1] = np.random.randint(10 - int(num_digit_list[0, 1]), 9)
            num_digit_list[1, 2] = np.random.randint(10 - int(num_digit_list[0, 2]), 9)
        elif op1 == '-':
            for i in range(num_range):
                num_digit_list[0, i] = np.random.randint(1, 8)
            num_digit_list[1, 0] = np.random.randint(0, int(num_digit_list[0, 1]))
            num_digit_list[1, 1] = np.random.randint(int(num_digit_list[0, 1]) + 1, 9)
            num_digit_list[1, 2] = np.random.randint(int(num_digit_list[0, 0]) + 1, 9)
        num1 = 100 * num_digit_list[0, 0] + 10 * num_digit_list[0, 1] + num_digit_list[0, 2]
        num2 = 100 * num_digit_list[1, 0] + 10 * num_digit_list[1, 1] + num_digit_list[1, 2]
        for i in range(num_range):
            num_digit_list[2, i] = np.random.randint(1, 8)
        num3 = 100 * num_digit_list[2, 0] + 10 * num_digit_list[2, 1] + num_digit_list[2, 2]

        target = str(num1)+str(op1)+str(num2)+str(op2)+str(num3)
        result = self.cmds[op2](self.cmds[op1](num1, num2), num3)
        return target, result

    def low_level(self):
        num_range = 3
        num_digit_list = np.zeros([2, num_range])
        op = random.choice(self.ops)

        if op == '+':
            for i in range(num_range):
                num_digit_list[0, i] = np.random.randint(0, 9)
            num_digit_list[1, 0] = np.random.randint(0, 10 - int(num_digit_list[0, 0]))
            num_digit_list[1, 1] = np.random.randint(0, 10 - int(num_digit_list[0, 1]))
            num_digit_list[1, 2] = np.random.randint(0, 10 - int(num_digit_list[0, 2]))

        elif op == '-':
            for i in range(num_range):
                num_digit_list[0, i] = np.random.randint(1, 9)
            num_digit_list[1, 0] = np.random.randint(0, int(num_digit_list[0, 0]))
            num_digit_list[1, 1] = np.random.randint(0, int(num_digit_list[0, 1]))
            num_digit_list[1, 2] = np.random.randint(0, int(num_digit_list[0, 2]))

        num1 = 100 * num_digit_list[0, 0] + 10 * num_digit_list[0, 1] + num_digit_list[0, 2]
        num2 = 100 * num_digit_list[1, 0] + 10 * num_digit_list[1, 1] + num_digit_list[1, 2]

        target = str(num1) + str(op) + str(num2)
        result = self.cmds[op](num1, num2)
        return target, result

        # 第一步进入欢迎界面，介绍试验流程
    def create_stimulus(self, start_up_time=start_up_value):

        self.prompt_label1 = tk.Label(self, text="欢迎参加 心算 任务！")
        self.prompt_label1.pack(pady=10)

        self.prompt_label2 = tk.Label(self,
                                      text="点击开始后将开始后将进入{}s倒计时，倒计时结束的瞬间请开始记录试验数据".format(
                                          start_up_time), font=("Arial", 10))
        self.prompt_label2.pack(pady=10)

        self.word_label = tk.Label(self, text="", font=("Helvetica", 24))
        self.word_label.pack()

        self.entry = tk.Entry(self, width=20, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        self.check_button = tk.Button(self, text="检查答案", command=self.check_answer)
        self.check_button.pack()

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.difficultychoose_label = tk.Label(self, text="请选择难度")
        self.difficultychoose_label.pack(pady=10)

        self.next_button1 = tk.Button(self, text="简单", command=self.start_up_mixed)
        self.next_button1.pack(side=tk.BOTTOM, padx=10,pady=10)

        self.next_button2 = tk.Button(self, text="困难", command=self.start_up_mixed)
        self.next_button2.pack(side=tk.BOTTOM, padx=10,pady=10)

        self.next_button3 = tk.Button(self, text="混合", command=self.start_up_mixed)
        self.next_button3.pack(side=tk.BOTTOM, padx=10,pady=10)

    # 试验开始倒计时
    def start_up_mixed(self, remaining=start_up_value):
        if remaining > 0:
            self.prompt_label1.config(text="实验将在{}秒后开始".format(remaining))
            self.next_button1.destroy()
            self.next_button2.destroy()
            self.next_button3.config(text="", command=lambda: None)
            self.after(1000, self.start_up_mixed, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.start = self.sti
            self.MentalCalculation_task_mixed()

    def check_answer(self):
        self.user_input = self.entry.get().lower()
        if self.user_input == self.target_result:
            self.result_label.config(text="回答正确！", fg="green")
        else:
            self.result_label.config(text="回答错误！正确答案是" + self.target_result, fg="red")

    def rest_time_mixed(self, remaining=rest_time_value):

        if remaining > 0:
            self.prompt_label1.config(text="休息时间")
            self.prompt_label2.config(text="实验将在{}秒后开始".format(remaining))
            self.word_label.config(text="")
            self.result_label.config(text="")
            self.next_button3.config(text="", command=lambda: None)
            self.after(1000, self.rest_time_mixed, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.trail_num -= 1
            self.MentalCalculation_task_mixed()

    def MentalCalculation_task_mixed(self):
        self.end = time.perf_counter()
        self.prompt_label1.config(text="第{}轮测试".format(self.trail_max - self.trail_num))
        self.prompt_label2.config(text="请计算下式")

        self.result_label.config(text="")
        if ((self.end - self.sti) < self.task_time) & (self.trail_num > 0) & ((self.trail_max - self.trail_num) % 2 == 1):
            self.target_number, self.target_result = self.high_level()
            self.word_label.config(text=self.target_number)
            self.next_button3.config(text="下一个", command=self.MentalCalculation_task_mixed)
        elif ((self.end - self.sti) < self.task_time) & (self.trail_num > 0) & ((self.trail_max - self.trail_num) % 2 == 0):
            self.target_number, self.target_result = self.low_level()
            self.word_label.config(text=self.target_number)
            self.next_button3.config(text="下一个", command=self.MentalCalculation_task_mixed)
        elif ((self.end - self.sti) >= self.task_time) & (self.trail_num > 0):
            self.mark = self.end - self.start
            self.listT.append(self.mark)
            self.rest_time_mixed()
        elif self.trail_num <= 0:
            self.sav_list()
            self.destroy()


class DigitSpanExperiment(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("DigitSpanExperiment")
        self.geometry("500x500")
        self.create_stimulus()
        self.ops = ['+', '-']
        self.trail_max = trail_value + 1
        self.trail_num = trail_value
        self.task_time = task_time_value
        self.forward_span_length = forward_span
        self.backward_span_length = backward_span
        self.listT = []

    def sav_list(self):
        with open('DigitSpan_mark.txt', 'a', encoding='utf-8') as f:
            f.write('\n\n####################################################################################\n')
            f.write(str(datetime.datetime.now()))
            f.write('########################################################\n')
            for j in self.listT:
                f.write(str(j) + str('    '))
            f.write('\n####################################################################################\n\n\n\n\n')
            f.close()

    def generate_sequence(self, length):
        """
        生成指定长度的随机数字序列
        """
        sequence = str()
        for _ in range(length):
            digit = random.randint(0, 9)
            sequence = sequence + str(digit)
        return sequence


        # 第一步进入欢迎界面，介绍试验流程
    def create_stimulus(self, start_up_time=start_up_value):

        self.prompt_label1 = tk.Label(self, text="欢迎参加 数字记忆广度 任务！")
        self.prompt_label1.pack(pady=10)

        self.prompt_label2 = tk.Label(self,
                                      text="点击开始后将开始后将进入{}s倒计时，倒计时结束的瞬间请开始记录试验数据".format(
                                          start_up_time), font=("Arial", 10))
        self.prompt_label2.pack(pady=10)

        self.word_label = tk.Label(self, text="", font=("Helvetica", 24))
        self.word_label.pack()

        self.entry = tk.Entry(self, width=20, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        self.check_button = tk.Button(self, text="检查答案", command=self.check_answer)
        self.check_button.pack()

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.next_button1 = tk.Button(self, text="开始", command=self.start_up)
        self.next_button1.pack(side=tk.BOTTOM, padx=10,pady=10)


    # 试验开始倒计时
    def start_up(self, remaining=start_up_value):
        if remaining > 0:
            self.prompt_label1.config(text="实验将在{}秒后开始".format(remaining))
            self.next_button1.config(text="", command=lambda: None)
            self.after(1000, self.start_up, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.start = self.sti
            self.Digit_Span()

    def check_answer(self):
        self.user_input = self.entry.get().lower()
        if self.user_input == self.target_number:
            self.result_label.config(text="回答正确！", fg="green")
        else:
            self.result_label.config(text="回答错误！正确答案是" + self.target_number, fg="red")

    def rest_time(self, remaining=rest_time_value):
        if remaining > 0:
            self.prompt_label1.config(text="休息时间")
            self.prompt_label2.config(text="实验将在{}秒后开始".format(remaining))
            self.word_label.config(text="")
            self.result_label.config(text="")
            self.next_button1.config(text="", command=lambda: None)
            self.after(1000, self.rest_time, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.trail_num -= 1
            self.Digit_Span()

    def aq_part(self):
        if (self.trail_max - self.trail_num) % 2 == 1:
            self.prompt_label2.config(text="请正向重复数字")
            self.next_button1.config(text="下一个", command=self.Digit_Span)
            self.word_label.config(text="")
        elif (self.trail_max - self.trail_num) % 2 == 0:
            self.prompt_label2.config(text="请反向向重复数字")
            self.next_button1.config(text="下一个", command=self.Digit_Span)
            self.word_label.config(text="")

    def Digit_Span(self):
        self.end = time.perf_counter()
        self.prompt_label1.config(text="第{}轮测试".format(self.trail_max - self.trail_num))
        self.result_label.config(text="")
        if ((self.end - self.sti) < self.task_time) & (self.trail_num > 0) & ((self.trail_max - self.trail_num) % 2 == 1):
            self.prompt_label2.config(text="请记忆数字")
            self.target_number = self.generate_sequence(self.forward_span_length)
            self.word_label.config(text=self.target_number)
            self.next_button1.config(text="下一个", command=self.aq_part)

        elif ((self.end - self.sti) < self.task_time) & (self.trail_num > 0) & ((self.trail_max - self.trail_num) % 2 == 0):
            self.prompt_label2.config(text="请记忆数字")
            self.target_number = self.generate_sequence(self.backward_span_length)
            self.word_label.config(text=self.target_number)
            self.next_button1.config(text="下一个", command=self.aq_part)

        elif ((self.end - self.sti) >= self.task_time) & (self.trail_num > 0):
            self.mark = self.end - self.start
            self.listT.append(self.mark)
            self.rest_time()

        elif self.trail_num <= 0:
            self.sav_list()
            self.destroy()

class BreathExperiment(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("BreathExperiment")
        self.geometry("500x500")
        self.create_stimulus()
        self.ops = ['+', '-']
        self.trail_max = trail_value + 1
        self.trail_num = trail_value
        self.task_time = task_time_value
        self.forward_span_length = forward_span
        self.backward_span_length = backward_span
        self.listT = []

    def sav_list(self):
        with open('Breath.txt', 'a', encoding='utf-8') as f:
            f.write('\n\n####################################################################################\n')
            f.write(str(datetime.datetime.now()))
            f.write('########################################################\n')
            for j in self.listT:
                f.write(str(j) + str('    '))
            f.write('\n####################################################################################\n\n\n\n\n')
            f.close()


        # 第一步进入欢迎界面，介绍试验流程
    def create_stimulus(self, start_up_time=start_up_value):

        self.prompt_label1 = tk.Label(self, text="欢迎参加 呼吸循环抑制 任务！")
        self.prompt_label1.pack(pady=10)

        self.prompt_label2 = tk.Label(self,
                                      text="点击开始后将开始后将进入{}s倒计时，倒计时结束的瞬间请开始记录试验数据".format(
                                          start_up_time), font=("Arial", 10))
        self.prompt_label2.pack(pady=10)


        self.next_button1 = tk.Button(self, text="开始", command=self.start_up)
        self.next_button1.pack(side=tk.BOTTOM, padx=10,pady=10)


    # 试验开始倒计时
    def start_up(self, remaining=start_up_value):
        if remaining > 0:
            self.prompt_label1.config(text="实验将在{}秒后开始".format(remaining))
            self.next_button1.config(text="", command=lambda: None)
            self.after(1000, self.start_up, remaining - 1)
        else:
            self.sti = time.perf_counter()
            self.start = self.sti
            self.Digit_Span()


    def rest_time(self, remaining=task_time_value):
        if remaining > 0:
            self.prompt_label1.config(text="不准呼吸")
            self.prompt_label2.config(text="实验将在{}秒后开始".format(remaining))

            self.next_button1.config(text="", command=lambda: None)
            self.after(1000, self.rest_time, remaining - 1)
        else:
            self.trail_num -= 1
            self.Digit_Span()

    def breath_time(self, remaining=rest_time_value):
        if remaining > 0:
            self.prompt_label1.config(text="第{}轮测试".format(self.trail_max - self.trail_num))
            self.prompt_label2.config(text="请呼吸，实验将在{}秒后开始".format(remaining))
            self.next_button1.config(text="", command=lambda: None)
            self.after(1000, self.breath_time, remaining - 1)
        else:
            self.rest_time()

    def Digit_Span(self):
        if self.trail_num > 0:
            self.breath_time()

        elif self.trail_num <= 0:
            self.sav_list()
            self.destroy()


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Lin's tasks")
        self.master.geometry("500x300+500+300")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        stroop_button = tk.Button(self, text="Stroop", command=self.open_stroop_experiment)
        stroop_button.pack(pady=10)

        mental_calculation_button = tk.Button(self, text="MentalCalculation", command=self.open_mental_calculation_experiment)
        mental_calculation_button.pack(pady=10)

        digit_span_button = tk.Button(self, text="Digit Span", command=self.digit_span_experiment)
        digit_span_button.pack(pady=10)

        breath_button = tk.Button(self, text="breath", command=self.breath_experiment)
        breath_button.pack(pady=10)

    def open_stroop_experiment(self):
        stroop_window = StroopExperiment(self.master)

    def open_mental_calculation_experiment(self):
        mental_calculation_window = MentalCalculationExperiment(self.master)

    def digit_span_experiment(self):
        mental_calculation_window = DigitSpanExperiment(self.master)

    def breath_experiment(self):
        mental_calculation_window = BreathExperiment(self.master)

if __name__ == '__main__':
    # 运行程序
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()