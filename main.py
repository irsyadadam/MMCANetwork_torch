import argparse
from dataload.dataload import load_blink_dataset



if __name__ == '__main__':

    ((m1_train_x, m1_train_y), (m1_test_x, m2_test_y)), ((m2_train_x, train_y), (m2_test_x, test_y)) = load_blink_dataset()


