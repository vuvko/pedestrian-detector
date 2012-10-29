#ifndef MODEL_H
#define MODEL_H

#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QImage>
#include <QtCore/qmath.h>
#include <QStringList>
#include <QTime>
#include <QPainter>
#include <QDebug>
#include <map>

class Model
{
public:
    Model();
    ~Model();

    bool saveModel();
    bool loadModel();
    void sampleAndTrain();
    void bootstrapTrain();
    void crossValidation();
    void classifyDirectory();
    bool estimateQuality();
    QImage scanImage(const QString &imageName);

    void setTrueLocationFileName(const QString &fileName);
    void setUserLocationFileName(const QString &fileName);
    void setModelFileName(const QString &fileName);
    void setDirName(const QString &dirName);
    void setKernel(bool useLinear);
    void setBound(double bound);
    void setBootstrap(bool useBootstrap);
    void setCV(bool useCV);

    QString getTrueLocationFileName();
    QString getUserLocationFileName();
    QString getModelFileName();    
    QString getDirName();
    double getPrecision();
    double getRecall();
    double getFScore();

    typedef QVector<double> Cell;

private:
    enum
    {
        CELL_NUM = 8,
        CELL_SIZE = 8,

        BLOCK_SIZE = 2,

        WIN_WIDTH = 80,
        WIN_HEIGHT = 200,
        WIN_WIDTH_CELL = 10,
        WIN_HEIGHT_CELL = 25,

        BACKGROUND_PER_IMG = 1,
        CROSS_PROC = 50,

        BOOTSTRAP_TIME = 25,
        BOOTSTRAP_BACK_PER_STEP = 100,
        BACK_STEP = 60,

        CV_TIME = 5
    };
    void sample();
    void trainModel();
    QVector<int> detect(const QImage &image);
    int defCell(qreal dx, qreal dy);
    QVector<QVector<Cell> > computeCells(const QImage &image);
    QVector<double> computeFeatures(const QVector<QVector<Cell> > &cells,
                                    int shift = 0);
    QVector<QVector<double> > generateFilter();
    void samplePositive();
    void sampleNegative();
    QVector<QImage> getBackgrounds();
    QVector<QImage> getPedestrians();
    QVector<double> convertFeatures(const QVector<double> &features);

    QVector< QVector<double> > m_features;
    QVector<int> m_labels;
    int m_featuresNumber;
    int m_instancesNumber;
    struct model* m_classifier;
    double m_precision;
    double m_recall;
    double m_fScore;
    QString m_trueLocationFileName;
    QString m_userLocationFileName;
    QString m_modelFileName;
    QString m_dirName;
    bool m_useLinear;
    bool m_useBootsTrap;
    bool m_useCV;
    int m_n;
    double m_L;
    double m_bound;
};

#endif // MODEL_H
