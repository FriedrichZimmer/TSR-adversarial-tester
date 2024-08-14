"""
Copyright (c) 2024 Friedrich Zimmer
creating a statistics based on the classifier results and exporting them as excel
statistics is directly based on the amount of images in a certain folder
"""
from argparse import ArgumentParser

import cv2

import os
import pandas as pd

CLASS_NOT_IN_CLASSIFIER = [
    ['No_Parking', 'deepranjang'],
    ['No_Parking', 'yashanksingh_v5'],
    ['KFC', 'deepranjang'],
    ['KFC', 'yashanksingh_v5'],
    ['KFC', 'again-3ibij'],
    ['Texaco', 'deepranjang'],
    ['Texaco', 'yashanksingh_v5'],
    ['Texaco', 'again-3ibij'],
    ['Speed_40', 'deepranjang'],
    ['Speed_40', 'yashanksingh_v5'],
    ['Speed_40', 'again-3ibij']
]

DEFAULT_PIXEL_MIN = 20


def category_not_in_classifier(sign, classifier, unknown_signs):
    """checks if a certain class can not be recognized by a classifier model, because the model has not been trained
    for this class"""
    for i in unknown_signs:
        if sign == i[0] and classifier[:-4] == i[1]:
            return True
    return False


def image_pixel_above_min(image_path, pixel_min):
    """ Checks if the size of an image is above a certain value
    """
    image = cv2.imread(image_path)
    try:
        h, w, c = image.shape
        if h < pixel_min or w < pixel_min:
            return False
        return True
    except AttributeError:
        print(f'Error when reading {image_path}')
        return False


def calc_classif_precision(good, below_threshold, adv, other):
    """ calculates precision, recall and the success rate of adversarial attacks
    Args:
        good: The amount of images, that have been classified correctly/true positive
        below_threshold: The amount of images where the classification confidence is below the defined threshold
        adv: The amount of images that have been classifies as the adversarial target
        (0 if image is not an adversarial example)
        other: The amount of images, that have been classified as another wrong category
    """

    # those calculations assume that alle cropped images were detected correctly
    if good + adv + other == 0:
        precision = -1
        success = 0
    else:
        # wrong classification is both a false negative and a false positive
        precision = 100 * good / (good + adv + other)
        success = 100 * adv / (good + adv + other)

    if good + adv + other + below_threshold > 0:
        recall = 100 * good / (good + adv + other + below_threshold)
    else:
        recall = -1

    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0

    return round(precision, 1), round(recall, 1), round(success, 1), round(f_score, 1)


def add_precision(df):
    """ Calculates precision, recall, adversarial sucess and F-Score and adds it to an existing dataframe as
    additional columns

    Args:
        df (DataFrame): pandas dataframe of a statistics lift including good, below_threshold, adversarial, other_bad

    Returns:
        The same dataframe with four more columns added
    """

    # add four more columns with the default value -1
    df['Precision_%'], df['Recall_%'], df['Adv_success_%'], df['F_score'] = [-1.0, -1.0, -1.0, -1.0]

    # fill columns
    for index, row in df.iterrows():
        precision, recall, success, f_score = (
            calc_classif_precision(row['good'], row['below_threshold'], row['adversarial'], row['other_bad']))
        df.at[index, 'Precision_%'] = precision
        df.at[index, 'Recall_%'] = recall
        df.at[index, 'Adv_success_%'] = success
        df.at[index, 'F_score'] = f_score
    return df


def get_signs_from_name(sign):
    """ It has been defined during image generation, that a # in a sign name
     is an adversarial sign in the format actualsign#adversarialsign
     This function retrieves the actual and adversarial signfrom the sign name

    Args:
        sign(str): Name of the sign during testing
     """

    if '#' in sign:
        actual_sign = sign[:sign.rfind('#')]
        adv_sign = sign[sign.rfind('#') + 1:]
    else:
        actual_sign = sign
        adv_sign = ''
    return actual_sign, adv_sign


class TsrStatistics:
    """calculates precision, recall and other statistics for all detector and classifier results in the test folder and
    exports them as excel"""

    def __init__(self, testproject_folder, not_learned_signs=None):
        """ initiates the statistics calculation

        Args:
            testproject_folder (str): Folder with the data that has to be analyzed
            pixel_min (int): Minimum pixels size. All images with x or y below this value are ignored for the statistics
            not_learned_signs ([str,str]): List of signs that are unknown for certain classifiers.
            Therefore below_threshold would be the true positive result for those cases
        """

        # initate test configuration
        self.testproject_folder = testproject_folder
        self.pixel_min = DEFAULT_PIXEL_MIN
        self.not_learned_signs = not_learned_signs

        # initiate variables
        self.result_table_detector = []
        self.result_table_classifier_raw = []
        self.result_table_classifier_enhanced = []

        self.num_detections = 0
        self.num_detections_pixel_min = 0

        self.actual_sign = ''
        self.adv_sign = ''

        self.sign = ''
        self.cam = ''
        self.detector = ''
        self.classifier = ''

        print('Statistic calculation initiated')

    def analyse_tests(self, pixel_min=DEFAULT_PIXEL_MIN):
        """ starts collecting data from the test folder
        """

        self.pixel_min = pixel_min
        self.result_table_detector = []
        self.result_table_classifier_raw = []
        self.result_table_classifier_enhanced = []
        print(self.not_learned_signs)

        print(f'Calculating Statistics for {self.testproject_folder}...')
        # cycling all test results. Each test is another traffic sign texture
        for self.sign in os.listdir(self.testproject_folder):
            test_folder = os.path.join(self.testproject_folder, self.sign)
            if os.path.isfile(test_folder):
                continue

            print(f'-Analyzing Sign {self.sign}...')

            # get actual/adversarial sign
            self.actual_sign, self.adv_sign = get_signs_from_name(self.sign)

            for self.cam in os.listdir(test_folder):
                print(f'--Analyzing Camera {self.cam}...')
                cam_folder = os.path.join(test_folder, self.cam)

                self.analyse_detectors(cam_folder)

        print('Statistic calculation finished')

    def analyse_detectors(self, cam_folder):
        """ classify all detection results """
        for self.detector in os.listdir(cam_folder):
            if self.detector == '_video':
                continue
            detector_folder = os.path.join(cam_folder, self.detector)
            if os.path.isfile(detector_folder):
                continue
            print(f'---Analyzing Detector {self.detector}...')
            cropped_folder = os.path.join(detector_folder, 'cropped')
            # get values for detectors
            self.num_detections = 0
            self.num_detections_pixel_min = 0
            self.count_detections(cropped_folder)
            self.result_table_detector.append([self.sign, self.cam, self.detector, self.num_detections,
                                               self.num_detections_pixel_min])

            # if not self.calc_class_stat:
            #     continue

            self.count_classifications(cropped_folder)

    def count_detections(self, cropped_folder):
        for cropped_image in os.listdir(cropped_folder):
            cropped_image_path = os.path.join(cropped_folder, cropped_image)
            if not os.path.isfile(cropped_image_path):  # skip the classifier folders
                continue
            self.num_detections += 1
            # value for 0
            if image_pixel_above_min(cropped_image_path, self.pixel_min):
                self.num_detections_pixel_min += 1

    def count_classifications(self, cropped_folder):
        # get values for classifiers
        for self.classifier in os.listdir(cropped_folder):
            classifier_folder = os.path.join(cropped_folder, self.classifier)
            if os.path.isfile(classifier_folder):  # skip the cropped images
                continue
            print(f'----Analyzing Classifier {self.classifier}...')

            good, below_threshold, adv, other = self.count_result_signs(classifier_folder)

            precision, recall, success, f_score = calc_classif_precision(good, below_threshold, adv, other)

            self.result_table_classifier_enhanced.append([self.sign,
                                                          self.actual_sign,
                                                          self.adv_sign,
                                                          self.cam,
                                                          self.detector,
                                                          self.classifier,
                                                          self.num_detections_pixel_min,
                                                          good,
                                                          below_threshold,
                                                          adv,
                                                          other,
                                                          precision,
                                                          recall,
                                                          success])

    def count_result_signs(self, classifier_folder):
        good = 0
        below_threshold = 0
        adv = 0
        other = 0

        # during classification all cropped images have been copied into subfolders named by category
        # now counting the amount of images in all subfolders
        for result_sign in os.listdir(classifier_folder):
            resultsign_folder = os.path.join(classifier_folder, result_sign)
            if not os.path.isfile(resultsign_folder) and result_sign != 'pp' and result_sign != '_video':
                # num_classifications is the amount of detections for a certain category
                num_classifications = len(os.listdir(resultsign_folder))
                num_classif_pix_above_min = 0
                for sorted_image in os.listdir(resultsign_folder):
                    sorted_image_path = os.path.join(resultsign_folder, sorted_image)
                    # For the statistics images below a certain size are ignored
                    if image_pixel_above_min(sorted_image_path, self.pixel_min):
                        num_classif_pix_above_min += 1

                self.result_table_classifier_raw.append([self.sign,
                                                         self.cam,
                                                         self.detector,
                                                         self.classifier,
                                                         self.num_detections_pixel_min,
                                                         result_sign,
                                                         num_classifications,
                                                         num_classif_pix_above_min])
                # print(f'{result_sign}, {self.actual_sign}, {self.adv_sign}')
                if result_sign == self.actual_sign:
                    good = num_classif_pix_above_min
                elif result_sign == '_below_threshold':
                    below_threshold = num_classif_pix_above_min
                elif result_sign == self.adv_sign:
                    adv = num_classif_pix_above_min
                else:
                    other += num_classif_pix_above_min

                # print(self.actual_sign, self.classifier)
                if category_not_in_classifier(self.actual_sign, self.classifier, self.not_learned_signs):

                    if good > 0:
                        print('ERROR: Sign actually exists in classifier!')
                    # if class is not in classifier below_threshold is the correct classification
                    good = below_threshold
                    below_threshold = 0
        # print(f'good: {good} {below_threshold} {adv} {other}')
        return good, below_threshold, adv, other

    def export_stat_excel(self, calc_class_stat=True, calc_camera_stat=True, cam_filter=None, detector_filter=None):
        """exports the results as excel

        Args:
            calc_class_stat (bool): If also the classifier stats should be exported or only the detector stats
            calc_camera_stat (bool): If also statistics for various cameras should be exported
            (unneeded if there is only one camera)
            detector_filter:
            cam_filter:
        """
        print('Exporting Result Excel...')
        tsr_stat_filename = f'tsr_stat_{self.pixel_min}'
        if cam_filter:
            tsr_stat_filename = f'{tsr_stat_filename}_{cam_filter}'
        if detector_filter:
            tsr_stat_filename = f'{tsr_stat_filename}_{detector_filter}'

        result_excel = pd.ExcelWriter(f'{self.testproject_folder}\\{tsr_stat_filename}.xlsx', mode='w')

        # predefining this header name as it is used in several places
        num_det_px = f'num_det_px>={self.pixel_min}'

        # calculate detector statistics
        # this statistics assumes that there are no false positive detections and no double detections.
        df_detector = pd.DataFrame(self.result_table_detector,
                                   columns=['Traffic Sign', 'Camera', 'Detector', 'num_detections',
                                            f'num_det_px>={self.pixel_min}'])
        if cam_filter:
            print(f'Camera == {cam_filter}')
            df_detector = df_detector.query('Camera == @cam_filter')
        if detector_filter:
            print(f'Detector == {detector_filter}')
            df_detector = df_detector.query('Detector == @detector_filter')
        df_detector.to_excel(result_excel, sheet_name='Detector_Results')

        # only shows sum of found images for each detector
        df_calc = df_detector[['Detector', 'num_detections', num_det_px]]
        df_calc = df_calc.groupby(['Detector']).sum()

        df_calc.to_excel(result_excel, sheet_name='Detector_Efficiency')

        # detection robustness by sign
        df_calc = df_detector[['Traffic Sign', 'Detector', 'num_detections', num_det_px]]
        df_calc = df_calc.groupby(['Traffic Sign', 'Detector']).sum()
        df_calc.to_excel(result_excel, sheet_name='Detector_TrafficSign')

        # detection robustness by camera
        if calc_camera_stat:
            df_calc = df_detector[['Camera', 'Detector', 'num_detections', num_det_px]]
            df_calc = df_calc.groupby(['Camera', 'Detector']).sum()
            df_calc.to_excel(result_excel, sheet_name='Detector_Camera')

        # calculate classifier statistics
        if calc_class_stat:
            # calculate classifier statistics
            df_raw = pd.DataFrame(self.result_table_classifier_raw,
                                  columns=['Traffic Sign', 'Camera', 'Detector', 'Classifier',
                                           num_det_px,
                                           'Result sign', 'num_classifications', f'num_class_px>={self.pixel_min}'])
            if cam_filter:
                df_raw = df_raw.query('Camera == @cam_filter')
            if detector_filter:
                df_raw = df_raw.query('Detector == @detector_filter')
            # export excel
            df_raw.to_excel(result_excel, sheet_name='Class_Raw_Data')

            df_results = pd.DataFrame(self.result_table_classifier_enhanced,
                                      columns=['Traffic Sign', 'Actual sign', 'Advers sign', 'Camera', 'Detector',
                                               'Classifier', num_det_px, 'good', 'below_threshold',
                                               'adversarial', 'other_bad',
                                               'precision (%)', 'recall (%)', 'adv_success (%)'])
            if cam_filter:
                df_results = df_results.query('Camera == @cam_filter')
            if detector_filter:
                df_results = df_results.query('Detector == @detector_filter')
            df_results.to_excel(result_excel, sheet_name='Class_Result_Details')

            # general classification robustness
            df_calc = df_results[
                ['Detector', 'Classifier', num_det_px, 'good', 'below_threshold', 'adversarial', 'other_bad']]
            df_calc = df_calc.groupby(['Detector', 'Classifier']).sum()
            df_calc = add_precision(df_calc)
            df_calc.to_excel(result_excel, sheet_name='Class_Robustness')

            # classification robustness by sign
            df_calc = df_results[
                ['Traffic Sign', 'Detector', 'Classifier', num_det_px, 'good', 'below_threshold', 'adversarial',
                 'other_bad']]
            df_calc = df_calc.groupby(['Traffic Sign', 'Detector', 'Classifier']).sum()
            df_calc = add_precision(df_calc)
            df_calc.to_excel(result_excel, sheet_name='Class_TrafficSign')

            # classification robustness against various camera conditions
            if calc_camera_stat:
                df_calc = df_results[['Detector', 'Classifier', 'Camera', num_det_px, 'good', 'below_threshold',
                                      'adversarial', 'other_bad']]
                df_calc = df_calc.groupby(['Detector', 'Classifier', 'Camera']).sum()
                df_calc = add_precision(df_calc)
                df_calc.to_excel(result_excel, sheet_name='Class_Camera')

            # classification robustness without attacks
            df_calc = df_results[df_results['Advers sign'] == '']
            df_calc = df_calc[
                ['Detector', 'Classifier', num_det_px, 'good', 'below_threshold', 'adversarial', 'other_bad']]
            df_calc = df_calc.groupby(['Detector', 'Classifier']).sum()
            df_calc = add_precision(df_calc)
            df_calc.to_excel(result_excel, sheet_name='Class_Basic_Robustness')

            # classification robustness_attacks_only
            df_calc = df_results[df_results['Advers sign'] != '']
            df_calc = df_calc[
                ['Detector', 'Classifier', num_det_px, 'good', 'below_threshold', 'adversarial', 'other_bad']]
            df_calc = df_calc.groupby(['Detector', 'Classifier']).sum()
            df_calc = add_precision(df_calc)
            df_calc.to_excel(result_excel, sheet_name='Class_Attack_Robustness')

            if calc_camera_stat:
                df_calc = df_results[
                    ['Detector', 'Classifier', 'Camera', num_det_px, 'good', 'below_threshold', 'adversarial',
                     'other_bad']]
                df_calc = df_calc.groupby(['Detector', 'Classifier', 'Camera']).sum()
                df_calc = add_precision(df_calc)
                df_calc.to_excel(result_excel, sheet_name='Class_Attack_Camera')

            result_excel.close()

        return result_excel


def simple_statistics(testproject_folder):
    """Wrapper for creating a simple statistics with default values"""
    tsr_stats = TsrStatistics(testproject_folder, CLASS_NOT_IN_CLASSIFIER)
    tsr_stats.analyse_tests(DEFAULT_PIXEL_MIN)
    os.startfile(tsr_stats.export_stat_excel())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    parser.add_argument('-camfilter', type=str, help='statisitics only for a single cam')
    parser.add_argument('-detectorfilter', type=str, help='statisitics only for a single detector')
    parser.add_argument('-p', type=int, help='Minimum Image size in pixel to be analysed',
                        default=DEFAULT_PIXEL_MIN)

    args = parser.parse_args()
    tsr_statistics = TsrStatistics(args.testproject_folder, CLASS_NOT_IN_CLASSIFIER)
    tsr_statistics.analyse_tests(args.p)
    os.startfile(tsr_statistics.export_stat_excel(cam_filter=args.camfilter, detector_filter=args.detectorfilter))
