import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycine.raw import read_frames
import skimage
from skimage.filters import threshold_otsu, sobel, try_all_threshold, threshold_local, threshold_triangle, threshold_minimum
import cv2 as cv
import re
import glob



def find_files(pattern):
    """Use regex to locate desired files
    Returns list of file paths.
    """
    cine_files = glob.glob(r"D:\Raw_DPSP-Data/*/*.cine")
    pattern = re.compile(pattern)
    matches = [file for file in cine_files if pattern.search(file) is not None]
    
    return matches


def load_cine(file, start_frame=0, n_frames=1):
    """Transforms cine file into np.array."""
    cine_file = file
    start_frame = start_frame
    count = n_frames
    raw_images, setup, bpp = read_frames(cine_file, start_frame=start_frame, count=count)
    images = np.array([next(raw_images) for _ in range(n_frames)],dtype=np.float32)
    
    return images


def array_norm(image):
    im = image.copy()
    im_max = im.max()
    im_min = im.min()
    im_norm = (im-im_min)/(im_max-im_min)

    return im_norm


def thresh_image(im_norm, thresh=0.05):
    """sets all pixels lower than threshold to 1 and everything else to zero.
    """

    im_norm = im_norm.copy()
    im_norm_max = im_norm.max()
    im_norm[im_norm > thresh] = im_norm_max*2 # arb number used as intermediate stage
    im_norm[im_norm <= thresh] = 1
    im_norm[im_norm == im_norm_max*2] = 0
    
    return im_norm


def process_image(img_raw):
    """get image ready for cv2 circle finder."""

    img = array_norm(img_raw)
    # create multiple images at different thresholds then compile into one image
    im_thresh_lst = []
    for i in np.arange(0.005,0.1,0.005):
        im_thresh = thresh_image(img, thresh=i)
        im_thresh_lst.append(im_thresh)
    labs = np.sum(im_thresh_lst, axis=0)
    # blur out noise
    img = skimage.filters.gaussian(labs, sigma=2, )
    img = array_norm(img)
    # cv takes 8 bit image
    img = img*100 
    img = img.astype(np.uint8) 
    
    return img


def find_distance(coords):
    """Calculate distances between list of coordinates
    input: list of n tuples of shape (2,)
    output: matrix nxn of distances"""
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    
    x, y = coords.T
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    distance = np.sqrt((x-x.T)**2 + (y-y.T)**2)
    return distance


def img_dict(file_path):
    """Returns a dict of same image in different formats."""
    ims = load_cine(file_path)
    img_raw = ims[0,:,:]
    processed_image = process_image(img_raw)
    img_raw = img_raw/np.quantile(img_raw, 0.99) 
    display_img = cv.cvtColor(img_raw,cv.COLOR_GRAY2BGR)
    return {
        'raw_image': img_raw,
        'processed_image': processed_image, 
        'display_image': display_img,
    }

class SetUpLabelledDataset:
    
    # add circle gen that allows user to change hough circle params
    def _find_circles(self, file_path=None, param1 = 25, param2 = 15):
        
        images = img_dict(file_path)
        processed_image = images['processed_image']
        cimg = images['display_image'].copy()
        while True:
            circles = cv.HoughCircles(
                processed_image,cv.HOUGH_GRADIENT,1,50,
                param1=param1,param2=param2, minRadius=0,maxRadius=10
            )
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                for circle in circles:
                    cv.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),2)
                plt.imshow(cimg)
                plt.show(block=False)
                print("Add or remove circles? add/remove:")
                more_less = input()
                if more_less == 'add':
                    return self._find_circles(file_path, param1 = 25, param2 = param2 - 5)
                elif more_less == 'remove':
                    return self._find_circles(file_path, param1 = 25, param2 = param2 + 5)
                else:
                    plt.close()
                    del images
                    del cimg
                    return {'param1': param1, 'param2': param2}


 


    def create_labelled_dataset(self, file_paths: dict, labelled_datset_file_path):
        # Data that will be used to automate labelling
        # HoughCricle params tried and tested
        
        labelled_configs_array = []
        for config, path in file_paths.items():
            circle_params = self._find_circles(path)
            images = img_dict(path)
            processed_image = images['processed_image']
            display_image = images['display_image']
            
            circles = cv.HoughCircles(
                processed_image,cv.HOUGH_GRADIENT,1,50,
                param1=circle_params['param1'],
                param2=circle_params['param2'], 
                minRadius=0,maxRadius=10
            )
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                circles_labelled = {}
                labels = []
                for i, circle in enumerate(circles):
                    sing_circ_img = display_image.copy()
                    # draw the outer circle
                    cv.circle(sing_circ_img,(circle[0],circle[1]),circle[2],(0,255,0),2)
                    plt.imshow(sing_circ_img)
                    plt.show(block=False, )
                    print(f"Enter the pressure tap name of captured by the circle (enter 'na' if not a pressure tap):")
                    pressure_tap_name = input()
                    already_exists = pressure_tap_name in labels
                    not_allowed = pressure_tap_name in ['x', 'y', 'r'] 
                    if already_exists or not_allowed :
                        if already_exists: print('Label already exists. try again:')
                        else: print("Tap cannot be labelled: 'x', 'y', or 'r'. Use another label:")
                        fixed = False
                        while not fixed:
                            pressure_tap_name = input()
                            if pressure_tap_name not in labels:
                                fixed = True
                            else:
                                print('Label already exists. try again:')

                    if pressure_tap_name != 'na':
                        labels.append(pressure_tap_name)
                        circles_labelled[i] = pressure_tap_name
                        plt.close()
                    
                print("Labelling complete.")
                ## Add something here to allow an edit

                all_circs_df = pd.DataFrame(circles, columns=['x', 'y', 'r'])
                ptap_labels_df = pd.DataFrame.from_dict(circles_labelled, orient='index')   
                labelled_df = all_circs_df.join(ptap_labels_df).dropna().reset_index(drop=True)
                labelled_df.rename(columns={0:'ptap'}, inplace=True)
                labelled_df['configuration'] = config

                # add circle distances (distance from each ptap)
                circle_coords = [*zip(labelled_df['x'], labelled_df['y'])]
                distance_mat = find_distance(circle_coords)            
                dist_df = pd.DataFrame(np.array(distance_mat).reshape(labelled_df.shape[0],labelled_df.shape[0]))
                dist_df.columns = labelled_df.ptap
                labelled_df = labelled_df.join(dist_df)
                labelled_df = self.check_labels(images['display_image'], labelled_df)

                labelled_configs_array.append(labelled_df)
                del images['raw_image']
                del images['processed_image'] 
                del images['display_image']
                del sing_circ_img
        labelled_configs_df = pd.concat(labelled_configs_array, axis=0)
        labelled_configs_df.to_csv(labelled_datset_file_path, index=False)
        print("CSV file created.")


    
    def check_labels(self, cimg=None, labelled_df=None):
        ax = plt.subplot()
        ax.imshow(cimg)
        for i in range(labelled_df.shape[0]):
            ax.text(labelled_df.x[i], labelled_df.y[i], labelled_df.ptap[i], c='r', size=15)
        plt.show(block=False,)
        print("Are the labels correct? (y/n):")
        inp = input()
        if inp == 'y': 
            plt.close()
            return labelled_df
        
        while True:
            print("Enter incorrect tap label:")
            incorrect_name = input()
            print("Correct label:")
            correct_name = input()
            labelled_df.loc[labelled_df.ptap == incorrect_name, 'ptap'] = correct_name
            print("anymore? (y/n):")
            inp = input()
            if inp == 'n':
                plt.close()
                return labelled_df
        
     



class PressureTapLabeller:
    """Automatated labelling of pressure taps"""
    def __init__(self, labelled_datset_csv_path):
        self.labelled_dataset = pd.read_csv(labelled_datset_csv_path)
        
         
    def _find_circles_and_dists(self, img_processed, max_dist_between_circs=50, param1=25,param2=15, minRadius=0,maxRadius=10):
        """
        Searches image for circular items that meet specs
        Creates df containing circle position, radius and distances. 
        """

        circles_all = cv.HoughCircles(img_processed,cv.HOUGH_GRADIENT,1,max_dist_between_circs,
                                    param1=param1,param2=param2, minRadius=minRadius,maxRadius=maxRadius)
        if circles_all is not None:
            circles_all = np.uint16(np.around(circles_all[0]))
        circle_coords_df = pd.DataFrame(circles_all, columns=['x', 'y', 'r'])
        circle_coords_df = circle_coords_df.sort_values('x', ascending=False).reset_index(drop=True)

        circle_coords = [*zip(circle_coords_df['x'], circle_coords_df['y'])]
        # find distance between circles
        distance_mat = find_distance(circle_coords)
        dist_df = pd.DataFrame(np.array(distance_mat).reshape(len(circle_coords_df),len(circle_coords_df)))
        circle_pos_and_dist = circle_coords_df.join(dist_df)
        return circle_pos_and_dist


    def _score_circles(self, file, circle_pos_and_dist, verbose=False):
        """wkd smaht stuff"""
        # Scoring unlabelled circle distances against ptap distances to determine if circle is a ptap
        score_df = pd.DataFrame()
        
        ptap_df = self.get_ptap_df(file)

        only_ptap_columns = ~ptap_df.columns.isin(['x', 'y', 'r', 'ptap', 'configuration'])
        all_ptap_dists = ptap_df.loc[:,only_ptap_columns].dropna(axis=1, ) # remove ptaps that aren't found config

        # iterate through set of distances for each unlabelled circle
        for i in range(len(circle_pos_and_dist)):
            unlabelled_dists = np.around(circle_pos_and_dist.iloc[i, 4:].values.astype('float32'),1)
            if verbose:
                print(f"\nresults for index {i} tap\n==========================")

            # iterate through set of distances for each ptap
            for j in range(len(ptap_df)):  
                ptap_name = ptap_df.iloc[j].ptap
                ptap_dists = np.around(all_ptap_dists.iloc[j,:].values.astype('float32'), 1) 

                # create a bool matrix where unknown circle dist sets are compared to ptap dist sets.
                mask = np.isclose(unlabelled_dists.reshape(-1,1), ptap_dists, atol=5) # true if unlabelled distance matches ptap dist
                ptap_dists_broadcast = np.broadcast_to(ptap_dists, (len(unlabelled_dists),len(ptap_dists))) # need to broadcast to apply mask
                matches = set(ptap_dists_broadcast[mask]) # returns matching distances. duplicates removed through set 
                match_rating = round(100*len(matches)/len(ptap_dists),0)
                score_df.loc[i, ptap_name] = match_rating
                
                if verbose:
                    print(f"ptap {ptap_name} has a {match_rating}% match")

        # Add highest scoring ptap name and score to score df and then join with circle df
        # This labels each circle record with the most likely pressure tap and drops any circles that scored lower than threshold.
        highest_scoring_ptap = score_df.apply(lambda row: row.idxmax(), axis=1)
        highest_score = score_df.apply(lambda row: row.values.max(), axis=1)
        score_df['highest_scoring_ptap'] = highest_scoring_ptap
        score_df['highest_score'] = highest_score
        print(score_df)
        print('================')
        print(circle_pos_and_dist)
        predicted_ptap_and_score = score_df[['highest_scoring_ptap', 'highest_score']]
     
        match_df = predicted_ptap_and_score.join(circle_pos_and_dist)[score_df.highest_score>70] # remove any circles with a score less than 70
        match_df['config'] = file
        match_df['config'] = match_df.config.str.strip(r'D:\Raw_DPSP-Data\\')
        print(match_df)
        
        return match_df


    def get_ptap_df(self, file):
        """Returns the correct ptap info df depending on test configuration."""
        
        ptap_df = self.labelled_dataset
        if re.search("Empty-Bay", file) is not None:
            file_ptap_df = ptap_df.loc[ptap_df['configuration']=='no_store']
        elif re.search("Store-in", file) is not None:
            file_ptap_df = ptap_df.loc[ptap_df['configuration']=='store_in']
        elif re.search("Store-out", file) is not None:
            file_ptap_df = ptap_df.loc[ptap_df['configuration']=='store_out']
        
    #     display(file_ptap_df.head())
        return file_ptap_df

    def show_ptap_assignment(self, cimg, matched_circles):
        """Plots images showing processing carried out to identify ptaps."""
        cimg2 = cimg.copy()
        circles_filtered = matched_circles.loc[:,['x', 'y', 'r']].values

        for i in circles_filtered:
            cv.circle(cimg2,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
            cv.circle(cimg2,(i[0],i[1]),2,(255,0,0),3)
        
        kwargs = {'aspect':'equal'}
        ax = plt.subplot()
        ax.imshow(cimg2) 
        ax.invert_xaxis()
        # annotate labelled filtered circle plot
        ptap_coords = [*zip(matched_circles.x, matched_circles.y)]
        for i, coord in enumerate(ptap_coords):
            ax.text(*coord, matched_circles.highest_scoring_ptap.values[i], c='r', size=20)

        plt.show()

    def find_ptaps(self, file_path_pattern):
        
        files = find_files(file_path_pattern)
        for file_path in files:
            print(file_path)
            ims = load_cine(file_path)
            img_raw = ims[0,:,:]
            processed_image = process_image(img_raw)
            display_img = img_raw/np.quantile(img_raw, 0.99)
            display_img = cv.cvtColor(display_img,cv.COLOR_GRAY2BGR)
            circle_info_df = self._find_circles_and_dists(processed_image)
            matched_circles = self._score_circles(file_path, circle_info_df, verbose=False)
            self.show_ptap_assignment(display_img, matched_circles)


if __name__ == '__main__':
    # no store
    no_store_pat=r"D:\\Raw_DPSP-Data\\62070-DPSP-Empty-Bay-doors-off-sawtooth-LE-TE-Beta0\\MBH009_62070_M_000_Beta_0_CineF24.cine"
    # store in
    store_in_pat=r"D:\Raw_DPSP-Data\62130-DPSP-Store-in-doors-on-sawtooth-LE-TE-Beta0\MBH009_62130_M-000-Beta-0_CineF5.cine"
    #store out
    store_out_pat=r"D:\Raw_DPSP-Data\62160-DPSP-Store-out-doors-on-sawtooth-LE-TE-Beta0\MBH009_62160_M-000-Beta_00_CineF22.cine"

    files = {
        'no_store': no_store_pat, 
        'store_in': store_in_pat, 
        'store_out': store_out_pat,
    }

    # SetUpLabelledDataset().create_labelled_dataset(file_paths=files, labelled_datset_file_path='labelled.csv')




    PressureTapLabeller('labelled.csv').find_ptaps('95')

    def get_ptap_luminosity_means(file, matched_circles_df, frame_count=100):
        """100 frames gives good results. no need to use more."""
        img_gen, _, _ = read_frames(file, start_frame=0, count=frame_count) # around 6600 frames for r'D:\Raw_DPSP-Data\62040-DPSP-Empty-Bay-doors-off-no-LE-TE-Beta0\MBH009_62040-M000_Beta_0_CineF10.cine'

        circles_filtered = matched_circles_df.loc[:,['x', 'y', 'r', 'highest_scoring_ptap']].values

        luminosity_df = pd.DataFrame()
        for i in range(frame_count):
            img_raw = np.array(next(img_gen), dtype=np.float32)
            cimg2 = cv.cvtColor(array_norm(img_raw),cv.COLOR_GRAY2BGR) 
            cimg2*=2.5
            # Iterate through each ptap -> create surrounding ring -> record luminosity at each ring pixel -> take mean -> add to df 
            for j, circle in enumerate(circles_filtered):
                cv.circle(cimg2,(circle[0],circle[1]),circle[2]+7,(0,255,0),4) # ring has inner radius of 7-4/2 and outer of 7+4/2
                ring_coords = np.argwhere(cimg2[:,:,1]==255)
                ring_luminosity = img_raw[ring_coords[:,0], ring_coords[:,1]]
                
                # Apply a threshold to remove any bad pixels that might be included
    #             e.g. => ring_luminosity_filtered = ring_luminosity[ring_luminosity>0]
                
                
                luminosity_df.loc[i, circle[-1]] = ring_luminosity.mean() # mean of each pixel
                ########## pretty sure this needs to be added so that frame is reset and ring isn't included in the next iteration.
                ######### show image to for each iter to see what happens.
    #             cimg2 = cv.cvtColor(array_norm(img_raw),cv.COLOR_GRAY2BGR) 
    #             cimg2*=2.5
                
            del img_raw ## Not sure why this is here


        luminosity_mean = luminosity_df.apply(lambda x: x.mean()) # mean of ptap across frames
        luminosity_mean_df = pd.pivot_table(luminosity_mean.reset_index(), columns='index')
        luminosity_mean_df['config'] = file
        
        return luminosity_df,luminosity_mean_df