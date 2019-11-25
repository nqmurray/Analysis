import pandas as pd
import numpy as np


"""
Functions used in the analysis notebooks. Kept in a seperate file both to maintain
the readability of the notebook (focus is the data, not the code) and also to make easier to
update for all notebooks.
"""
def import_datasets(file_list, normalize_data, norm_to_zero=False): # From a list of csv files import all data and add normalize y values if normalize_data is True.  Returns a DataFrame of all values from the imported files
    if normalize_data:
        frame_list = []
        # loop through all files and import data frames, normalize their y data and then concatenate full dataframe
        for f in file_list:
            df = pd.read_csv(f)
            # take column 1 (y data) and min max scale from -1 to 1
            if norm_to_zero is False:
                normed_data = df.iloc[:, 1].apply(lambda x: ((
                    ((x - df.iloc[:, 1].min()) / (df.iloc[:, 1].max() - df.iloc[:, 1].min())) - 0.5) * 2))
            else:  # take y data and normalize from 0-1
                normed_data = df.iloc[:, 1].apply(lambda x: (
                    (x - df.iloc[:, 1].min()) / (df.iloc[:, 1].max() - df.iloc[:, 1].min())))
            df.insert(2, 'Normalized ' + df.columns.values[1], normed_data)
            frame_list.append(df)
        # ignore_index = True makes concatentate axis index from 0 to n-1
        full_df = pd.concat((elem for elem in frame_list), ignore_index=True)
    else:
        full_df = pd.concat((pd.read_csv(f)
                             for f in file_list), ignore_index=True)
    return full_df


# drops indexed region via multiindex dictionary and key/values from dictionary
def drop_regions(df, ig_dict, graph_column_list, graph_column, hue_column):
    unfound = []  # list where items were not found
    ignored = []
    # iterate over items of dictionary and find their indexes in the df
    for key, value in ig_dict.items():
        if float(key) in graph_column_list:
            for elem in value:
                try:
                    # find indexes of df where key, value element match
                    indexes_to_drop = df[(df[graph_column] == float(key)) & (
                        df[hue_column] == elem)].index
                    ignored.append(df.loc[indexes_to_drop])
                    df.drop(indexes_to_drop, inplace=True)
                except Exception:
                    # append list with graph value and hue value not found string
                    unfound.append(
                        '{} graph {} hue values not found in dataframe'.format(key, elem))
        else:
            # append list with graph not found string
            unfound.append('{} graph not found in dataframe'.format(key))
    # ignored dataframe to plot for visual insprection
    if ignored != []:
        ignored_df = pd.concat((elem for elem in ignored), ignore_index=True)
    else:
        ignored_df = np.empty(0)
    if unfound == []:
        unfound.append('Drop regions successful.')

    return df, ignored_df, unfound


"""
Pass in a array, finds two innermost zeros (closest to the middle of the dataset)
and does a linear interpolation between the two points on either side of the zero and returns the x values found,
also returns flag if more than 2 zeros are found.  Will check for zeros that have the lowest index and the x value closest
to x average
"""
def find_zeros(xy_array, check_leftright):
    zero_list = find_zero_index(xy_array[:, 1], check_leftright)
    if len(zero_list) > 2:  # flag if more than 2 zeros are found
        extra_zero = 'True'
    else:
        extra_zero = 'False'
    # update zerolist to be just two points
    zero_list = find_closest(zero_list, xy_array[:, 0])
    x_list = []
    for ind_tup in zero_list:
        if ind_tup[1] == 0:
            x_list.append(xy_array[ind_tup[0], 0])  # x value at index
        else:
            i = (xy_array[ind_tup[0], 0], xy_array[ind_tup[0], 1])
            t = (xy_array[ind_tup[1], 0], xy_array[ind_tup[1], 1])
            # two points at index 1, index 2
            x_list.append(linear_interpolate_twopoints(i, t))

    x_list.sort(reverse=True)  # return list from pos to neg

    return x_list, extra_zero


# finds all zeros in the array, returns list of tuples that are the indexes of those points
def find_zero_index(x, points):  
    zero_list = []
    for index, elem in enumerate(x):
        if elem == 0 and compare_array_signs(x[index - points:index], x[index + 1:index + 1 + points]):
            # return zero point with index, zero as tuple
            zero_list.append((index, 0))
        elif compare_array_signs(x[index - points + 1:index + 1], x[index + 1:index + points + 1]):
            # return to points crossing zero as tuple
            zero_list.append((index, index + 1))
        else:
            pass
    return zero_list


# check arrays to see if points on both sides are of opposite signs
def compare_array_signs(left, right):  
    # check if array has values, avoids error with array being empty
    if left.size > 0 and right.size > 0 and np.sign(left) == -np.sign(right):
        return True  # there is the slim chance left and right are symmetric and so will erroneously return true
    else:
        return False

    
# finds the index of the two zeros (one per half of x_data length) with the lowest index that are closest to (x.max + x.min / 2)
def find_closest(index_list, x_data):
    # zero in the first half the dataset
    first_zero = (len(x_data), np.max(x_data))
    # zero in the second half in the dataset
    second_zero = (len(x_data), np.max(x_data))
    avg = (np.max(x_data) + np.min(x_data)) / 2
    for tup in index_list:
        # this method will erroneously grab zeros in low-high loops if data is really bad
        if tup[0] >= int(len(x_data) / 2):  # check if tuple is in second half
            if tup[0] <= second_zero[0] and abs(x_data[tup[0]] - avg) <= second_zero[1]:
                # check if index is lower than currently stored value and that distance from x_avg is less than stored value
                second_zero = tup
            else:
                pass
        else:
            if tup[0] <= first_zero[0] and abs(x_data[tup[0]] - avg) <= first_zero[1]:
                first_zero = tup
            else:
                pass

    return [first_zero, second_zero]


# linear interpolation between two known points passed in as tuples (x,y)
def linear_interpolate_twopoints(p1, p2):
    # given by equation: x = x1 - y1 * (x2-x1)/(y2-y1)
    return (p1[0] - p1[1] * ((p2[0] - p1[0]) / (p2[1] - p1[1])))



#  ==================================== PULSE SWITCHING  & AMR/USMR ANALYSIS =================================== #

# finds the difference between min and max average values
def find_resistance_change(xy_array, data_percent):
    # search through and find all indexes that are within data_percent % of abs(x max)
    x_array_index = [index for index, x in enumerate(xy_array[:, 0]) if
                     abs(x) >= abs(np.min(xy_array[:, 0])) - abs((np.min(xy_array[:, 0])) * data_percent / 100)]
    y_max = np.zeros(int(len(x_array_index) / 2) + 1)
    y_min = np.zeros(int(len(x_array_index) / 2) + 1)
    min_ind = 0
    max_ind = 0
    # find all y values for x indexes and assign them to min and max arrays
    for xy in xy_array[x_array_index, :]:
        if xy[0] > 0:
            y_max[max_ind] = xy[1]
            max_ind += 1
        else:
            y_min[min_ind] = xy[1]
            min_ind += 1
    # strip outliers and return the average, don't use zeros (extra length in arrays)
    high = average_without_outliers([y for y in y_max if y != 0])
    low = average_without_outliers([y for y in y_min if y != 0])

    return abs(high - low)  # difference between high and low


# discard all data that is beyond certain standard deviation, return the mean of remaining data
def average_without_outliers(data, m=2):
    return np.average([d for d in data if abs(d - np.mean(data)) < m * np.std(data)])


#  ==================================== AMR/USMR ANALYSIS =================================== #

 # x, y, normed y data, finds peaks and associated y values
def find_peaks(xyn_array): 
    half = int(len(xyn_array[:, 1]) / 2)
    max1_index = np.argmax(xyn_array[:half, 1])  # check first half
    # check second half, account for it being second half
    max2_index = np.argmax(xyn_array[half:, 1]) + half
    max1_x = xyn_array[max1_index, 0]
    max2_x = xyn_array[max2_index, 0]
    if max1_x > max2_x:
        # return right x peak, left x peak, avg x, right y peak, left y peak, right normed peak, left normed peak
        return [max1_x, max2_x, (max1_x + max2_x) / 2, xyn_array[max1_index, 1],
                xyn_array[max2_index, 1], xyn_array[max1_index, 2], xyn_array[max2_index, 2]]
    else:
        return [max2_x, max1_x, (max1_x + max2_x) / 2, xyn_array[max2_index, 1],
                xyn_array[max1_index, 1], xyn_array[max2_index, 2], xyn_array[max1_index, 2]]
