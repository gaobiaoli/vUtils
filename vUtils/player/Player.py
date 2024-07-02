import cv2


class Player:
    def __init__(self, window_name="Player", width=900, height=600, shotFPS=False):
        self.window_name = window_name
        self.player_inited = False
        self.writer_inited = False
        self.width = width
        self.height = height
        self.shotFPS = shotFPS

    def show_FPS(self):
        # TODO: showFPS in Video
        self.show_FPS = True

    def init_player(self, window_name, width, height):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    def init_writer(self, filename, FPS, width, height):
        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        self.output = cv2.VideoWriter(filename, fourcc, FPS, (width, height))
        self.writer_inited = True
        return self.output

    def show(self, img):
        if not self.player_inited:
            self.init_player(self.window_name, self.width, self.height)
            self.player_inited = True

        cv2.imshow(self.window_name, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow(self.window_name)
            exit()

    def write(self, img):
        if not self.writer_inited:
            self.init_writer(
                filename="output.avi", FPS=25, width=img.shape[1], height=img.shape[0]
            )
        self.output.write(img)

    def showImg(self, img):
        if not self.player_inited:
            self.init_player(self.window_name, self.width, self.height)
            self.player_inited = True
        cv2.imshow(self.window_name, img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyWindow(self.window_name)
            exit()

    def release(self):
        if self.writer_inited:
            self.output.release()
        cv2.destroyAllWindows()
