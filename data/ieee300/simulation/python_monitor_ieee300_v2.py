# -*- coding: utf-8 -*-
import os
import time
import subprocess
import signal

if __name__ == '__main__':
    file_path = 'E:\\simulation\\code\\psse3304_tutorials\\logs\\log300_v2.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('')
        f.close()
    script_path = 'E:\\simulation\\code\\psse3304_tutorials\\ieee300\\generate_freq_data300_v2.py'
    check_interval = 30
    timeout = 180
    process = subprocess.Popen(['python', script_path])
    last_modified = os.path.getmtime(file_path)
    timeout_start = time.time()

    while True:
        time.sleep(check_interval)
        try:
            current_modified = os.path.getmtime(file_path)
        except Exception as e:
            print(e)
            break

        if current_modified > last_modified: # 文件有更新，重置最后修改时间和超时时间
            last_modified = current_modified
            timeout_start = time.time()
        else:
            # 文件没有更新，检查是否超时
            if time.time() - timeout_start > timeout:
                print("文件长时间未更新，重启脚本")
                # 终止当前进程
                process.terminate()
                process.wait()
                # 重启脚本
                process = subprocess.Popen(['python', script_path])
                # 重置最后修改时间和超时时间
                last_modified = os.path.getmtime(file_path)
                timeout_start = time.time()
