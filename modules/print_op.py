
class bcolors:
    HEADER = '\033[95m'  # 粉红色
    OKBLUE = '\033[94m'  # 蓝色
    OKGREEN = '\033[92m'  # 绿色
    WARNING = '\033[93m'  # 黄色
    FAIL = '\033[91m'  # 红色
    ENDC = '\033[0m'  # 默认
    UNDERLINE = '\033[4m'  # 下划线


def print_FAIL(text):
    # 输出红色字体
    print(bcolors.FAIL + str(text) + bcolors.ENDC)


def print_WARNING(text):
    # 输出黄色字体
    print(bcolors.WARNING + str(text) + bcolors.ENDC)


def print_BLUE(text):
    # 输出蓝色字体
    print(bcolors.OKBLUE + str(text) + bcolors.ENDC)


def print_GREEN(text):
    # 输出绿色字体
    print(bcolors.OKGREEN + str(text) + bcolors.ENDC)
