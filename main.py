from kivy.app import App

from ui.mainScreen import MainScreen



class SnapPrintApp(App):

    def build(self):
        self.mainScreen = MainScreen()

        return self.mainScreen



    def on_stop(self):
        self.mainScreen.StopCamera()



if __name__ == "__main__":
    SnapPrintApp().run()