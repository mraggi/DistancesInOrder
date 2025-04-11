from fastprogress import progress_bar as fpb

class VariableSizeProgressBar(fpb):
    def on_update(self, val, text):
        self.total = len(self.gen)
        super().on_update(val,text)

progress_bar = VariableSizeProgressBar

