from DcgmiDiag import DcgmiDiag

def main():
    dd = DcgmiDiag(dcgmiPrefix='.')
    passedCount = 0
    for i in range(0, 160):
        print "&&&& RUNNING dcgmi_diag_test"
        failed = dd.Run()
        if failed:
            print "&&&& FAILED dcgmi_diag_test"
            dd.PrintLastRunStatus()
        else:
            print "&&&& PASSED dcgmi_diag_test"
            passedCount += 1

if __name__ == '__main__':
    main()