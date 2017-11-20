import matplotlib.pyplot as plt

class Drawer():

    def __init__(self,y_pred, y_test, target_names, i,X_test,eigenfaces):
        self.y_pred=y_pred
        self.y_test=y_test
        self.target_names=target_names
        self.i=i
        self.X_test=X_test

    def plot_gallery(self,images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())


    # plot the result of the prediction on a portion of the test set

    def title(self):
        pred_name = self.target_names[self.y_pred[self.i]].rsplit(' ', 1)[-1]
        true_name = self.target_names[self.y_test[self.i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


    def show(self,h,w):
        prediction_titles = [self.title() for i in range(self.y_pred.shape[0])]

        self.plot_gallery(self.X_test, prediction_titles, h, w)

        # plot the gallery of the most significative eigenfaces

        eigenface_titles = ["eigenface %d" % i for i in range(self.eigenfaces.shape[0])]
        self.plot_gallery(self.eigenfaces, eigenface_titles, h, w)
        plt.show()
