import numpy as np
import PySimpleGUI as sg


# 定义功能
def array_operations(operation, a=None, b=None, params=None):
    try:
        if operation == "数组形状判断":
            return a.shape
        elif operation == "数组形状改变":
            new_shape = tuple(params)
            return a.reshape(new_shape)
        elif operation == "全零数组创建":
            shape = tuple(params)
            return np.zeros(shape)
        elif operation == "全一数组创建":
            shape = tuple(params)
            return np.ones(shape)
        elif operation == "单位矩阵生成":
            n = params[0]  # 矩阵行数
            m = (params[1] if len(params) > 1 else n)  # 矩阵列数，默认和行数相同
            return np.eye(n, m)
        elif operation == "随机数组生成":
            shape = tuple(params)
            return np.random.random(shape)
        elif operation == "数组加法运算":
            return a + b
        elif operation == "数组减法运算":
            return a - b
        elif operation == "数组乘法运算":
            return a * b
        elif operation == "数组除法运算":
            return a / b
        elif operation == "数组点积运算":
            return np.dot(a, b)
        elif operation == "转置点积运算":
            return np.dot(a, a.T)
        elif operation == "数组转置运算":
            return a.T
        elif operation == "数组元素求和":
            return np.sum(a)
        elif operation == "数组平均值计算":
            return np.mean(a)
        elif operation == "数组标准差计算":
            return np.std(a)
        elif operation == "数组最小值查找":
            return np.min(a)
        elif operation == "数组最大值查找":
            return np.max(a)
        elif operation == "最小值下标查找":
            return np.argmin(a)
        elif operation == "最大值下标查找":
            return np.argmax(a)
        elif operation == "数组升序排序":
            return np.sort(a)
        elif operation == "数组拼接":
            return np.concatenate((a, b))
        elif operation == "数组拆分":
            split_num = params[0]  # 拆分出的子数组数量
            return np.array_split(a, split_num)
        elif operation == "均匀间隔数组创建":
            start, stop, num = params  # start为起始元素值，stop为结束元素值，num为元素个数
            return np.linspace(start, stop, num)
        elif operation == "指定大小过滤数组":
            num = params[0]
            return a[a > num]
        elif operation == "复制数组":
            return np.copy(a)
        elif operation == "数组元素加指定值":
            num = params[0]
            return a + np.array(num)
        elif operation == "数组元素求平方根":
            return np.sqrt(a)
        elif operation == "数组元素求平方":
            return np.square(a)
        elif operation == "数组元素对数运算":
            return np.log(a)
        elif operation == "数组元素指数运算":
            return np.exp(a)
        elif operation == "数组数据类型查看":
            return a.dtype
        elif operation == "获取数组大小":
            return a.size
        elif operation == "获取数组字节数":
            return a.nbytes
        elif operation == "平铺数组":
            times = tuple(params)
            return np.tile(a, times)
        elif operation == "数组降维为一":
            return a.flatten()
        elif operation == "获取逆矩阵":
            return np.linalg.inv(a)
        elif operation == "计算方阵行列式":
            return np.linalg.det(a)
        elif operation == "计算特征值和特征向量":
            eigenvalues, eigenvectors = np.linalg.eig(a)
            return eigenvalues, eigenvectors
    except Exception as e:
        return str(e)


# 定义功能选项和对应的注释
Operations = {
    "数组形状判断": ["array"],
    "数组形状改变": ["new_shape"],
    "全零数组创建": ["shape"],
    "全一数组创建": ["shape"],
    "单位矩阵生成": ["N", "M"],
    "随机数组生成": ["shape"],
    "数组加法运算": [],
    "数组减法运算": [],
    "数组乘法运算": [],
    "数组除法运算": [],
    "数组点积运算": [],
    "转置点积运算": [],
    "数组转置运算": [],
    "数组元素求和": [],
    "数组平均值计算": [],
    "数组标准差计算": [],
    "数组最小值查找": [],
    "数组最大值查找": [],
    "最小值下标查找": [],
    "最大值下标查找": [],
    "数组升序排序": [],
    "数组拼接": [],
    "数组拆分": ["split_num"],
    "均匀间隔数组创建": ["start", "stop", "num"],
    "指定大小过滤数组": ["num"],
    "复制数组": [],
    "数组元素加指定值": ["num"],
    "数组元素求平方根": [],
    "数组元素求平方": [],
    "数组元素对数运算": [],
    "数组元素指数运算": [],
    "数组数据类型查看": [],
    "获取数组大小": [],
    "获取数组字节数": [],
    "平铺数组": ["times"],
    "数组降维为一": [],
    "获取逆矩阵": [],
    "计算方阵行列式": [],
    "计算特征值和特征向量": ["eigenvalues", "eigenvectors"]
}

# 创建GUI布局
layout = [
    [sg.Text("功能选择"), sg.Combo([op for op in Operations.keys()], size=(20, 10), key="OPERATION")],
    [sg.Text("请输入数组a(以分号分隔行，逗号分隔列)"), sg.InputText(key="ARRAY_A")],
    [sg.Text("请输入数组b(以分号分隔行，逗号分隔列)"), sg.InputText(key="ARRAY_B")],
    [sg.Text("其他参数(多个则以逗号分隔)"), sg.InputText(key="PARAMS")],
    [sg.Button("运行"), sg.Button("退出")],
    [sg.Text("输出结果:")],
    [sg.Multiline(size=(80, 20), key="OUTPUT")]
]

# 创建窗口
window = sg.Window("Numpy数组运算器", layout)

# 事件循环
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "退出":
        break

    if event == "运行":
        operation = values["OPERATION"]
        try:
            # 解析输入的数组a
            array_a = None
            if values["ARRAY_A"]:
                array_a = np.array([list(map(float, row.split(','))) for row in values["ARRAY_A"].split(';')])

            # 解析输入的数组b
            array_b = None
            if values["ARRAY_B"]:
                array_b = np.array([list(map(float, row.split(','))) for row in values["ARRAY_B"].split(';')])

            # 解析额外参数
            params = None
            if values["PARAMS"]:
                params = list(map(int, values["PARAMS"].split(',')))

            result = array_operations(operation, array_a, array_b, params)
            if isinstance(result, tuple):
                result = f"{result[0]},{result[1]}"
            window["OUTPUT"].update(result)
        except Exception as e:
            window["OUTPUT"].update(f"Error: {e}")
window.close()
