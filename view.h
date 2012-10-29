#ifndef VIEW_H
#define VIEW_H

#include <QWidget>
#include <QMessageBox>
#include <QFileDialog>
#include <iview.h>
#include <QEvent>
#include <QDebug>

namespace Ui
{
    class View;
}

class View : public QWidget, public IView
{
    Q_OBJECT

public:
    explicit View(QWidget *parent = 0);
    ~View();

    void setViewPrecision(double value);
    void setViewRecall(double value);
    void setViewFScore(double value);
    void setViewScannedImage(const QImage &image);
    void showWarning(const QString &warningMsg);

signals:
    void setTrueLocationFileName(const QString &fileName);
    void setUserLocationFileName(const QString &fileName);
    void setModelFileName(const QString &fileName);
    void setCurrDir(const QString &dirName);
    void setKernel(bool useLinear);
    void setBound(double bound);
    void saveModel();
    void loadModel();
    void train();
    void bootstrap();
    void crossValidation();
    void classifyDirectory();
    void estimateQuality();
    void scanImage(const QString &fileName);

private slots:
    void chooseDirButton();
    void chooseTrueLocationFileButton();
    void chooseUserLocationFileButton();
    void chooseTCModelFileButton();
    void chooseSIModelFileButton();
    void chooseImageForScan();
    void trainButton();
    void classifyDirectoryButton();
    void estimateQualityButton();
    void scanImageButton();

private:
    Ui::View *ui;
    QMessageBox *warningDialog;
    QLabel *imageLabel;

    double m_precision;
    double m_recall;
    double m_fScore;
};

#endif // VIEW_H
