from get_source_arr import get_source_arr, pre_cal
from blur_cal import blur_cal
from get_time_arr import get_time_arr

ball_boxes, net_boxes = get_source_arr()
# print(ball_boxes, net_boxes)
print('hello word')

finialResults = pre_cal(ball_boxes, net_boxes)

finialResults = blur_cal(finialResults)

start_indices, end_indices = get_time_arr(finialResults)

def convert_arrays_to_json(start_indices, end_indices):
  result = []

  for index, i in enumerate(start_indices):
      obj = {
          'start': format(i, '.1f'),
          'end': format(end_indices[index], '.1f'),
          'name': ''
      }
      result.append(obj)
  print(result)
  return result

convert_arrays_to_json(start_indices, end_indices)

