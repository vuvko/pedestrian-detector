#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>
#include "model.h"

class IView;

class Controller: public QObject
{
    Q_OBJECT

public:
    Controller(IView* view);
    ~Controller();

private:
    Model* m_model;
    IView* m_view;

    void RefreshViewFull() const;
    void RefreshViewPRF() const;

private slots:
    void OnSetTrueLocationFileName(const QString &fileName);
    void OnSetUserLocationFileName(const QString &fileName);
    void OnSetModelFileName(const QString &fileName);
    void OnSetCurrDir(const QString &dirName);
    void OnSetKernel(bool useLinear);
    void onSetBound(double bound);

    void OnSaveModel();
    void OnLoadModel();
    void OnTrain();
    void OnBootstrap();
    void onCV();
    void OnClassifyDirectory();
    void OnEstimateQuality();
    void OnScanImage(const QString &fileName);
};

#endif // CONTROLLER_H

