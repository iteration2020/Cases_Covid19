# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from streamlit.web import cli as stcli
from streamlit import runtime
import sys
import regression_new

if __name__ == '__main__':
    if runtime.exists():
        regression_new.mainmenu()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
