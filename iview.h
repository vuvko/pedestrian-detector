#ifndef IVIEW_H
#define IVIEW_H

#include <QImage>

class IView
{
public:
    virtual void setViewPrecision(double value) = 0;
    virtual void setViewRecall(double value) = 0;
    virtual void setViewFScore(double value) = 0;
    virtual void setViewScannedImage(const QImage &image) = 0;
    virtual void showWarning(const QString &warningMsg) = 0;

public: // signals
    virtual void setTrueLocationFileName(const QString &fileName) = 0;
    virtual void setUserLocationFileName(const QString &fileName) = 0;
    virtual void setModelFileName(const QString &fileName) = 0;
    virtual void setCurrDir(const QString &dirName) = 0;
    virtual void setKernel(bool useLinear) = 0;
    virtual void setBound(double bound) = 0;
    virtual void saveModel() = 0;
    virtual void loadModel() = 0;
    virtual void train() = 0;
    virtual void bootstrap() = 0;
    virtual void crossValidation() = 0;
    virtual void classifyDirectory() = 0;
    virtual void estimateQuality() = 0;
    virtual void scanImage(const QString &fileName) = 0;
};

#endif // IVIEW_H
