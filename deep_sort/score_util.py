from tensorflow.python.ops.gen_array_ops import empty
import util.distance_util as du

# flag = 0 # default = 0; green. 1 = yellow, 2 = red.

def time_score(time):

    # default 30 fps
    seconds = int(time/10)

    if seconds < 30:
        timescore = 0.5

    elif seconds < 50:
        timescore = 1.5

    elif seconds < 100:
        timescore = 2

    else:
        timescore = 5
    
    # print("T SCORE : {}".format(timescore))
    return timescore


def distance_score(sea_line, road_line, box):

    center_x = (box[0] + box[2])/2
    center_y = (box[1] + box[3])/2
    distance_rate = 1/(du.near_score(sea_line, road_line,(center_x,center_y)))

    if distance_rate < 1:
        distancescore = 0.2

    elif distance_rate > 4:
        distancescore = 3
    
    elif distance_rate > 2:
        distancescore = 1    

    else:
        distancescore = distance_rate/2

    # print("D SCORE : {}".format(distancescore))

    return distancescore


def action_score(action, d_flag):

    action_list = {"1.Stand": 1, "2.Walk": 0.1, "4.SitDown": 5, "6.Ride": 0.1, "9.ClimbOver": 10, "10.Suicide": 100, "13.Fence": 5}

    default = 0

    if action in action_list:
        for act_class, score in action_list.items():
            if action == act_class:
                if action == ("9.ClimbOver" or "10.Suicide" or "13.Fence") and d_flag < 1:
                    return 1
                else:
                    # print("c {} s {}, action SCORE : {}".format(action,act_class,score))
                    return score
    
    else:
        return default

def score(track_time, seg_res, action_box, class_text):
    
    #print("INPUT VALUES : {}, {}, {}, {}".format(track_time, seg_res, action_box, class_text))

    t_score = time_score(track_time)
    

    sea_line, road_line = seg_res
   
    # sea_line, road_line = seg_res

    d_score = distance_score(sea_line, road_line, action_box)
    a_score = action_score(class_text, d_score)
    
    total_score = t_score * d_score * a_score

    # print("TOTAL SCORE : {}".format(total_score))

    return total_score


if __name__ == "__main__":
    try:
        print("score_util.py")
    except SystemExit:
        pass