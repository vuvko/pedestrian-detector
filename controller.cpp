#include "controller.h"
#include <iview.h>

Controller::Controller(IView* view)
{
    m_model = new Model();
    m_view = view;

    QObject* view_obj = dynamic_cast<QObject*>(m_view);

    QObject::connect(view_obj, SIGNAL(setTrueLocationFileName(const QString&)),
                     this, SLOT(OnSetTrueLocationFileName(const QString&)));
    QObject::connect(view_obj, SIGNAL(setUserLocationFileName(const QString&)),
                     this, SLOT(OnSetUserLocationFileName(const QString&)));
    QObject::connect(view_obj, SIGNAL(setModelFileName(const QString&)),
                     this, SLOT(OnSetModelFileName(const QString&)));
    QObject::connect(view_obj, SIGNAL(setCurrDir(const QString&)),
                     this, SLOT(OnSetCurrDir(const QString&)));
    QObject::connect(view_obj, SIGNAL(saveModel()), this, SLOT(OnSaveModel()));
    QObject::connect(view_obj, SIGNAL(loadModel()), this, SLOT(OnLoadModel()));
    QObject::connect(view_obj, SIGNAL(train()), this, SLOT(OnTrain()));
    QObject::connect(view_obj, SIGNAL(classifyDirectory()), this, SLOT(OnClassifyDirectory()));
    QObject::connect(view_obj, SIGNAL(estimateQuality()), this, SLOT(OnEstimateQuality()));
    QObject::connect(view_obj, SIGNAL(scanImage(const QString&)),
                     this, SLOT(OnScanImage(const QString&)));
    QObject::connect(view_obj, SIGNAL(bootstrap()), this, SLOT(OnBootstrap()));
    QObject::connect(view_obj, SIGNAL(setKernel(bool)), this, SLOT(OnSetKernel(bool)));
    QObject::connect(view_obj, SIGNAL(setBound(double)), this, SLOT(onSetBound(double)));
    QObject::connect(view_obj, SIGNAL(crossValidation()), this, SLOT(onCV()));

    RefreshViewFull();
}

Controller::~Controller()
{
    delete m_model;
}

void Controller::OnSaveModel()
{
    if (!m_model->saveModel())
        m_view->showWarning("Can not save model file!");
}

void Controller::OnLoadModel()
{
    if (!m_model->loadModel())
        m_view->showWarning("Can not load model file!");
}

void Controller::OnTrain()
{
    m_model->sampleAndTrain();
}

void Controller::OnBootstrap()
{
    m_model->setBootstrap(true);
}

void Controller::onCV()
{
    m_model->crossValidation();
}

void Controller::OnClassifyDirectory()
{
    m_model->classifyDirectory();
}

void Controller::OnEstimateQuality()
{
    if (m_model->estimateQuality())
        RefreshViewPRF();
    else
        m_view->showWarning("Can not open annotation file!");
}

void Controller::OnScanImage(const QString &imageName)
{
    m_view->setViewScannedImage(m_model->scanImage(imageName));
}

void Controller::OnSetTrueLocationFileName(const QString &fileName)
{
    m_model->setTrueLocationFileName(fileName);
}

void Controller::OnSetUserLocationFileName(const QString &fileName)
{
    m_model->setUserLocationFileName(fileName);
}

void Controller::OnSetModelFileName(const QString &fileName)
{
    m_model->setModelFileName(fileName);
}

void Controller::OnSetCurrDir(const QString &dirName)
{
    m_model->setDirName(dirName);
}

void Controller::OnSetKernel(bool useLinear)
{
    m_model->setKernel(useLinear);
}

void Controller::onSetBound(double bound)
{
    m_model->setBound(bound);
}

void Controller::RefreshViewFull() const
{
    RefreshViewPRF();
}

void Controller::RefreshViewPRF() const
{
    m_view->setViewPrecision(m_model->getPrecision());
    m_view->setViewRecall(m_model->getRecall());
    m_view->setViewFScore(m_model->getFScore());
}
