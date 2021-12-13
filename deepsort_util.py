def get_tracking_list(tracker):
    track_ids = []
    track_bboxes=[]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 

        track_id = int(track.track_id)
        track_ids.append(track_id)
        bbox = track.to_tlbr()
        crop_box = (int(bbox[0]), int(bbox[1]), (int(bbox[2])-int(bbox[0])), (int(bbox[3])-int(bbox[1]))) # x y w h
        track_bboxes.append(crop_box) 

    return track_ids, track_bboxes