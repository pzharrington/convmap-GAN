import tensorflow as tf
import keras
import io
import plots
import numpy as np


def _filebuf_to_tf_summary_img(filebuf, name):
    #img = tf.expand_dims(tf.image.decode_png(filebuf.getvalue(), channels=4), 0)
    return tf.Summary.Image(encoded_image_string=filebuf.getvalue())



class TboardImg(keras.callbacks.Callback):

    def __init__(self, title):
        super().__init__()
        self.title = title

    def on_epoch_end(self, iternum, GAN, logs={}):
        out = io.BytesIO()
        val = None
        if self.title == 'genimg':
            plots.save_img_grid(GAN.genrtor, GAN.noise_vect_len, fname=out, Xterm=False, scale=GAN.cscale)
        elif self.title == 'pixhist':
            val = plots.pix_intensity_hist(GAN.val_imgs, GAN.genrtor, GAN.noise_vect_len, 
                                           scaling=GAN.datascale, fname=out, Xterm=False)
        out.seek(0)
        image = _filebuf_to_tf_summary_img(out, self.title)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.title, image=image)])
        writer = tf.summary.FileWriter(GAN.expDir+'logs/imgs')
        writer.add_summary(summary, iternum)
        writer.close()
        out.close()
        return val

class TboardScalars(keras.callbacks.Callback):
    
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, GAN, iternum, logs):
        writer = tf.summary.FileWriter(GAN.expDir+'logs/scalars')
        for scalar_name in logs:
            summary = tf.Summary(value=[tf.Summary.Value(tag=scalar_name, simple_value=logs[scalar_name])])
            writer.add_summary(summary, iternum)
        writer.close()

class TboardSigmas(keras.callbacks.Callback):
    
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
    
    def on_epoch_end(self, GAN, iternum, logs):
        for scalar_name in logs:
            writer = tf.summary.FileWriter(GAN.expDir+'logs/sigmas/'+scalar_name)
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, simple_value=logs[scalar_name])])
            writer.add_summary(summary, iternum)
            writer.close()


class TboardHists(keras.callbacks.Callback):

    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, GAN, iternum, layerdict):
        for lyrname, weights in layerdict.items():
            name = self.tag+'_'+lyrname.replace('_', '')
            writer = tf.summary.FileWriter(GAN.expDir+'logs/hists/'+name)
            hist = self._buildhisto(weights)
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
            writer.add_summary(summary, iternum) 
            writer.close()
    
    def _buildhisto(self, values):

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=1000)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        return hist






