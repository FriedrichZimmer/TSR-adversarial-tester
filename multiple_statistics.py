"""Copyright (c) 2024 Friedrich Zimmer
Generating different statiscs based on the same result folder"""

from argparse import ArgumentParser

from tools.statistics_for_complete_model import TsrStatistics, CLASS_NOT_IN_CLASSIFIER

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')

    args = parser.parse_args()
    tsr_statistics = TsrStatistics(args.testproject_folder, CLASS_NOT_IN_CLASSIFIER)
    tsr_statistics.analyse_tests(30)
    # statistics for all results
    tsr_statistics.export_stat_excel()
    # statistics only for default camera
    tsr_statistics.export_stat_excel(cam_filter='01_default_new')
    # statistics only for Bestbox Detector
    # (this way the classifiers can be directly compared for different cameras without detector influence)
    tsr_statistics.export_stat_excel(detector_filter='Bestbox_0.0')

    # influence of image size
    tsr_statistics.export_stat_excel(detector_filter='Bestbox_0.0', cam_filter='01_default_new')
    tsr_statistics.analyse_tests(20)
    tsr_statistics.export_stat_excel()
    tsr_statistics.export_stat_excel(detector_filter='Bestbox_0.0', cam_filter='01_default_new')
    tsr_statistics.analyse_tests(40)
    tsr_statistics.export_stat_excel()
    tsr_statistics.export_stat_excel(detector_filter='Bestbox_0.0', cam_filter='01_default_new')
