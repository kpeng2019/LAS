import subprocess

def main():
    # print(subprocess.Popen("echo pkw", shell=True, stdout=subprocess.PIPE).stdout.read())
    ps = subprocess.Popen("type lime_fi.txt", shell=True,stdout=subprocess.PIPE)
    subprocess.call("sk.py --text 30 --latex True --higher True", shell=True,stdin=ps.stdout)
    return 0



if __name__ == "__main__":
    main()