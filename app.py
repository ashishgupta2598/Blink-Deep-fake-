import falcon
from SmileBlink_main import BlinkSmile_detection_video

api = application = falcon.API()
api.add_route('/ai/vision/video/liveliness/passive/video', BlinkSmile_detection_video())

