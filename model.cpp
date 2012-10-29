#include <model.h>
#include <cstdlib>
#include "liblinear-1.8/linear.h"

double sech(double x)
{
    return 2 * qExp(x) / (1 + qExp(2 * x));
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

Model::Model()
{
    m_featuresNumber = 0;
    m_instancesNumber = 0;
    m_precision = 0.0;
    m_recall = 0.0;
    m_fScore = 0.0;
    m_dirName = "";
    m_trueLocationFileName = "";
    m_userLocationFileName = "";
    m_modelFileName = "";
    m_classifier = NULL;
    m_useLinear = true;
    m_bound = -0.0;
    m_n = 1;
    m_L = 0.27;
}

Model::~Model()
{
    if (m_classifier)
        free_and_destroy_model(&m_classifier);
}

void Model::setTrueLocationFileName(const QString& fileName)
{
    m_trueLocationFileName = fileName;
}

void Model::setUserLocationFileName(const QString& fileName)
{
    m_userLocationFileName = fileName;
}

void Model::setModelFileName(const QString& fileName)
{
    m_modelFileName = fileName;
}

void Model::setDirName(const QString& dirName)
{
    m_dirName = dirName;
}

void Model::setKernel(bool useLinear)
{
    m_useLinear = useLinear;
}

void Model::setBound(double bound)
{
    m_bound = bound;
}

void Model::setBootstrap(bool useBootstrap)
{
    m_useBootsTrap = useBootstrap;
}

void Model::setCV(bool useCV)
{
    m_useCV = useCV;
}

QString Model::getTrueLocationFileName()
{
    return m_trueLocationFileName;
}

QString Model::getUserLocationFileName()
{
    return m_userLocationFileName;
}


QString Model::getModelFileName()
{
    return m_modelFileName;
}

QString Model::getDirName()
{
    return m_dirName;
}

double Model::getPrecision()
{
    return m_precision;
}

double Model::getRecall()
{
    return m_recall;
}

double Model::getFScore()
{
    return m_fScore;
}

bool Model::saveModel()
{
    // Save model
    QByteArray ba = m_modelFileName.toLocal8Bit();
    if (save_model(ba.data(), m_classifier))
        return false;
    else
        return true;
}

bool Model::loadModel()
{
    // Load model
    QByteArray ba = m_modelFileName.toLocal8Bit();
    if ((m_classifier = load_model(ba.data())) == 0)
        return false;

    // Read number of features from model file
    QFile file(m_modelFileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;
    QTextStream in(&file);
    QString str;
    do
        in >> str;
    while (str.compare("nr_feature"));
    in >> m_featuresNumber;
    file.close();
    return true;
}

void Model::sample()
{
    /* Sample positive and negative examples for subsequent training.

       Put feature vector of i's example in m_features[i] and
       its label in m_labels[i] (+1 for positive and -1 for negative examples).

       Training image are contained in training directory(m_dirName).
    */

    samplePositive();

    sampleNegative();
}

void Model::trainModel()
{
    /* Training procedure.

       You can vary SVM regularization parameter C(param.C)
    */

    m_featuresNumber = m_features[0].size();
    m_instancesNumber = m_features.size();

    struct problem prob;
    prob.l = m_instancesNumber;
    prob.bias = -1;
    prob.n = m_featuresNumber;
    prob.y = Malloc(int, m_instancesNumber);
    prob.x = Malloc(struct feature_node *, m_instancesNumber);

    for (int i = 0; i < m_instancesNumber; i++)
    {
        prob.x[i] = Malloc(struct feature_node, (this->m_featuresNumber)+1);
        prob.x[i][m_featuresNumber].index = -1;
        for (int j = 0; j < m_featuresNumber; j++)
        {
            prob.x[i][j].index = j+1;
            prob.x[i][j].value = m_features[i][j];
        }
        prob.y[i] = m_labels[i];
    }

    struct parameter param;
    param.solver_type = L2R_L2LOSS_SVC_DUAL;
    param.C = 0.3;      // try to vary it
    param.eps = 1e-4;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    m_classifier = train(&prob, &param);
    destroy_param(&param);
    free(prob.y);
    for (int i = 0; i < m_instancesNumber; i++)
        free(prob.x[i]);
    free(prob.x);
}

void Model::sampleAndTrain()
{
    m_features.clear();
    m_labels.clear();
    sample();
    trainModel();
    if (m_useBootsTrap)
    {
        bootstrapTrain();
    }
}

void Model::bootstrapTrain()
{
    QVector<QImage> backgrounds = getBackgrounds();
    QVector<QImage> pedestrians = getPedestrians();
    QTime time = QTime::currentTime();
    qsrand((uint)time.msec());

    for (int i = 0; i < BOOTSTRAP_TIME; ++i)
    {
        QVector<int> sizes(2, 0);
        for (int j = 0; j < BOOTSTRAP_BACK_PER_STEP; ++j)
        {
            int rnd = (qrand() + 0.0) / RAND_MAX * backgrounds.size();
            QVector<int> shifts = detect(backgrounds[rnd]);
            sizes[1] += shifts.size();
            for (int k = 0; k < shifts.size(); ++k)
            {
                QImage badExample(backgrounds[rnd].copy(shifts[k], 0, WIN_WIDTH, WIN_HEIGHT));
                qDebug() << "Saved in:" << (QString("bad_examples/") + QString::number(i) +
                                            QString("_") + QString::number(j) +
                                            QString("_") + QString::number(k) + QString("_back.png"));
                badExample.save(QString("bad_examples/") + QString::number(i) +
                                QString("_") + QString::number(j) +
                                QString("_") + QString::number(k) + QString("_back.png"));

                m_features.push_back(computeFeatures(computeCells(badExample)));
                m_labels.push_back(-1);
            }
        }
        for (int j = 0; j < pedestrians.size(); ++j)
        {
            QVector<int> shifts = detect(pedestrians[j]);
            if (shifts.size() != 0)
            {
                ++sizes[0];
                qDebug() << "Saved in:" << (QString("bad_examples/") + QString::number(i) +
                                            QString("_") + QString::number(j) + QString("_") +
                                            QString::number(0) + QString("_ped.png"));
                pedestrians[j].save(QString("bad_examples/") + QString::number(i) +
                                    QString("_") + QString::number(j) + QString("_") +
                                    QString::number(0) + QString("_ped.png"));

                m_features.push_back(computeFeatures(computeCells(pedestrians[j])));
                m_labels.push_back(1);
            }
        }
        if (sizes[0] > sizes[1])
        {
            int size = sizes[0] - sizes[1];
            for (int j = 0; j < size; ++j)
            {
                int rnd = (qrand() + 0.0) / RAND_MAX * backgrounds.size();
                int x = (qrand() + 1.0) / RAND_MAX * (backgrounds[rnd].width() - WIN_WIDTH_CELL - 1);
                m_features.push_back(computeFeatures(computeCells(backgrounds[rnd].copy(
                                                                      x, 0,
                                                                      WIN_WIDTH,
                                                                      WIN_HEIGHT))));
                m_labels.push_back(-1);
            }
        } else {
            int size = sizes[1] - sizes[0];
            for (int j = 0; j < size; ++j)
            {
                int rnd = (qrand() + 0.0) / RAND_MAX * pedestrians.size();
                m_features.push_back(computeFeatures(computeCells(pedestrians[rnd])));
                m_labels.push_back(1);
            }
        }

        trainModel();
    }
}

void Model::crossValidation()
{
    /*
    m_features.clear();
    m_labels.clear();
    sample();
    int errors = 0;
    QVector<QVector<double> > all_features = m_features;
    QVector<int> all_labels = m_labels;
    qDebug() << "Using Cross-Validation with parameters:";
    qDebug() << "Bound:" << m_bound;
    qDebug() << "n:" << m_n;
    qDebug() << "L:" << m_L;
    for(int i = 0; i < CV_TIME; ++i)
    {
        sparseFeatures(all_features, all_labels, i);
        trainModel();
        if (m_useBootsTrap)
        {
            bootstrapTrain();
        }

        QVector<std::pair<QVector<double>, int> > check = getCheck(all_features, all_labels, i);
        struct feature_node* x = Malloc(struct feature_node, this->m_featuresNumber + 1);
        x[m_featuresNumber].index = -1;
        for (int j = 0; j < check.size(); ++j)
        {
            for (int k = 0; k < m_featuresNumber; ++k)
            {
                x[k].index = k+1;
                x[k].value = check[j].first[k];
            }
            double prob_estimate;
            predict_values(m_classifier, x, &prob_estimate);

            double label = -1;
            if (prob_estimate > m_bound)
                label = 1;

            if (label != check[j].second)
                ++errors;
        }
    }
    qDebug() << "Cross-Validation finished. Errors:" << errors;
    */
}

QVector<int> Model::detect(const QImage &image)
{
    /* Detect objects in image using sliding window technique.

       You should return coordinates of all detected objects in QVector<int>.

       Follow instructions in code.
    */
    QVector<int> areas;
    QVector<QVector<Cell> > cells = computeCells(image);
    QVector<double> buff(cells[0].size() - WIN_WIDTH_CELL, 0);

    struct feature_node* x = Malloc(struct feature_node, this->m_featuresNumber + 1);
    x[m_featuresNumber].index = -1;
    //        slide with window along image and for each considering
    //        region do the following
    for (int i = 0; i < cells[0].size() - WIN_WIDTH_CELL; ++i)
    {
        //        compute feature vector of considering region and
        //        put it in 'descriptor'
        QVector<double> descriptor = computeFeatures(cells, i);

        for (int j = 0; j < m_featuresNumber; j++)
        {
            x[j].index = j+1;
            x[j].value = descriptor[j];
        }

        double prob_estimates[1];  // level of confidence
        int predicted_label = predict_values(m_classifier, x, prob_estimates);

        if (prob_estimates[0] > m_bound)
        {
            //areas.push_back(i * CELL_SIZE);
            buff[i] = prob_estimates[0];
        }
    }
    free(x);
    double max = 0;
    do
    {
        max = 0;
        int max_i = -2 * WIN_WIDTH_CELL;
        for (int i = 0; i < buff.size(); ++i)
        {
            if (buff[i] > max)
            {
                max = buff[i];
                max_i = i;
            }
        }
        if (max_i > 0)
        {
            areas.push_back(max_i * CELL_SIZE);
        }
        for (int j = qMax(max_i - WIN_WIDTH_CELL * CROSS_PROC / 100, 0);
             j < qMin(max_i + 1 + WIN_WIDTH_CELL * CROSS_PROC / 100, buff.size());
             ++j)
        {
            buff[j] = 0;
        }
    } while (max > 0);

    return areas;
}

void Model::classifyDirectory()
{
    /* Detect objects on images in directory(m_dirName).

       Write positions of detected objects in file(m_userLocationFileName)
       in form "<filename> <topLeftX> <topLeftY> <width> <height>\n".

       Use Model::detect method to find all objects in image.
    */

    QFile file(m_userLocationFileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        return;
    }
    qDebug() << m_userLocationFileName;
    QTextStream out(&file);

    QStringList img_filters;
    img_filters << "*.png" << "*.jpg" << "*.bmp" << "*.jpeg";
    QDir imgDir(m_dirName);
    imgDir.setNameFilters(img_filters);
    QStringList images = imgDir.entryList(QDir::NoFilter, QDir::Name);

    for (int i = 0; i < images.size(); ++i)
    {
        qDebug() << "File #" << i;
        QImage image(m_dirName + images.at(i));
        QVector<int> shifts = detect(image);

        QString fileName = images.at(i);
        fileName = fileName.left(fileName.lastIndexOf('.'));

        for (int j = 0; j < shifts.size(); ++j)
        {
            out << fileName << ' ' <<
                   shifts[j] << ' ' <<
                   0 << ' ' << WIN_WIDTH << ' ' <<
                   WIN_HEIGHT << '\n';
            qDebug() << fileName << shifts[j] << 0 << WIN_WIDTH << WIN_HEIGHT;
        }
    }

    file.close();
}

bool Model::estimateQuality()
{
    /* Estimate quality of classifier

       You should compute precision, recall and F-score(2*P*R/(P+R)) and
       put these values in m_precision, m_recall and m_fScore, respectively.
       Return true if everything is OK, false - otherwise.

       You can read predicted locations from m_userLocationFileName file and
       ground truth locations from m_trueLocationFileName file.
    */

    QFile trainedFile(m_userLocationFileName);
    QFile realFile(m_trueLocationFileName);
    if (!trainedFile.open(QIODevice::ReadOnly | QIODevice::Text) ||
            !realFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return false;
    }

    QTextStream trainedIn(&trainedFile), realIn(&realFile);
    QString fileName;
    int shift;

    std::map<QString, QVector<int> > trained, real;

    qDebug() << "Reading files...";

    while (!trainedIn.status())
    {
        trainedIn >> fileName >> shift;
        trainedIn.readLine();

        trained[fileName].push_back(shift);
    }

    while (!realIn.status())
    {
        realIn >> fileName >> shift;
        realIn.readLine();

        real[fileName].push_back(shift);
    }

    std::map<QString, QVector<int> >::const_iterator trIter = trained.begin();
    std::map<QString, QVector<int> >::const_iterator rlIter = real.begin();

    QVector<int> sizes(2, 0);
    for (trIter++; trIter != trained.end(); trIter++)
    {
        sizes[0] += trIter->second.size();
    }
    trIter = trained.begin();
    for (rlIter++; rlIter != real.end(); rlIter++)
    {
        sizes[1] += rlIter->second.size();
    }
    rlIter = real.begin();

    int tp = 0, tp1 = 0;
    int prevShift = -2 * WIN_WIDTH;
    QString prevName = "";

    qDebug() << "Estimating...";


    for (trIter++, rlIter++;
         trIter != trained.end() && rlIter != real.end();
         trIter++, rlIter++)
    {
        while (QString::compare(trIter->first, rlIter->first) < 0 && trIter != trained.end())
        {
            trIter++;
        }
        if (trIter == trained.end())
        {
            break;
        }
        while (QString::compare(trIter->first, rlIter->first) > 0 && rlIter != real.end())
        {
            rlIter++;
        }
        if (rlIter == real.end())
        {
            break;
        }
        int i = 0, j = 0;
        QVector<int> trShifts = trIter->second;
        QVector<int> rlShifts = rlIter->second;
        while (i < trShifts.size() && j < rlShifts.size())
        {
            while (trShifts[i] < rlShifts[j] - WIN_WIDTH * CROSS_PROC / 100 &&
                   i < trShifts.size())
            {
                ++i;
            }
            if (i >= trShifts.size())
            {
                break;
            }
            while (qAbs(trShifts[i] - rlShifts[j]) <= WIN_WIDTH * CROSS_PROC / 100 &&
                   i < trShifts.size() && j < rlShifts.size())
            {
                if (QString::compare(prevName, trIter->first) ||
                        qAbs(trShifts[i] - prevShift) > WIN_WIDTH * CROSS_PROC / 100)
                {
                    ++tp1;
                    prevShift = trShifts[i];
                    prevName = trIter->first;
                }
                ++tp;
                ++i;
                ++j;
            }
            if (i >= trShifts.size() || j >= rlShifts.size())
            {
                break;
            }
            while (trShifts[i] > rlShifts[j] + WIN_WIDTH * CROSS_PROC / 100 &&
                   j < rlShifts.size())
            {
                ++j;
            }
        }
    }

    qDebug() << "True positive:" << tp;
    qDebug() << "True positive':" << tp1;
    qDebug() << "Total trained:" << sizes[0];
    qDebug() << "Total real:" << sizes[1];

    m_precision = double(tp1) / sizes[0];
    m_recall = double(tp) / sizes[1];
    m_fScore = 2 * m_recall * m_precision / (m_recall + m_precision);

    return true;
}

QImage Model::scanImage(const QString &imageName)
{
    /* Scan single image for subsequent display.

       You should find all objects on imageName image and mark
       them with rectangles. Return marked image.

       Use Model::detect method to find all objects in image.
    */

    QImage image(imageName);
    QPainter paint(&image);
    paint.setPen(QColor(0, 255, 0));
    paint.setBrush(Qt::NoBrush);

    QVector<int> shifts = detect(image);

    for (int i = 0; i < shifts.size(); ++i)
    {
        paint.drawRect(shifts[i], 0, WIN_WIDTH, WIN_HEIGHT);
    }

    return image;
}

int Model::defCell(qreal dx, qreal dy)
{
    qreal direction = qAtan2(dx, dy);
    if (dx < 0)
    {
        direction += 2 * M_PI;
    }
    return floor(direction * 4 / M_PI);
}

QVector<QVector<Model::Cell> > Model::computeCells(const QImage &image)
{
    QVector<QVector<int> > grads(image.width(), QVector<int>(image.height(), 0));
    QVector<QVector<qreal> > filter = generateFilter();
    QVector<QVector<int> > buffer(image.width(), QVector<int>(image.height(), 0));

    for (int y = 0; y < image.height(); ++y)
    {
        QRgb *row = (QRgb *)image.scanLine(y);
        for (int x = 0; x < image.width(); ++x)
        {
            QRgb rgb = row[x];
            int grayLvl = 0.2125 * qRed(rgb) + 0.7154 * qGreen(rgb) +
                    0.0721 * qBlue(rgb);
            buffer[x][y] = grayLvl;
        }
    }

    for (int y = 1; y < image.height() - 1; ++y)
    {
        for (int x = 1; x < image.width() - 1; ++x)
        {
            double dx = 0, dy = 0;
            for (int i = 0; i < filter.size(); ++i)
            {
                for (int j = 0; j < filter[0].size(); ++j)
                {
                    int grayLvl = buffer[x - 1 + i][y - 1 + j];
                    dx += filter[i][j] * grayLvl;
                    dy += filter[j][i] * grayLvl;
                }
            }
            grads[x][y] = defCell(dx, dy);
        }
    }
    grads[0][0] = 0;
    grads[0][image.height() - 1] = 0;
    grads[image.width() - 1][0] = 0;
    grads[image.width() - 1][image.height() - 1] = 0;

    QVector<QVector<Cell> > cells(image.height() / CELL_SIZE,
                                  QVector<Cell>(image.width() / CELL_SIZE,
                                                Cell(CELL_SIZE, 0)));
    for (int i = 0; i < cells.size(); ++i)
    {
        for (int j = 0; j < cells[0].size(); ++j)
        {
            for (int dy = 0; dy < CELL_SIZE; ++dy)
            {
                int y = i * CELL_SIZE + dy;
                for (int dx = 0; dx < CELL_SIZE; ++dx)
                {
                    int x = j * CELL_SIZE + dx;
                    ++cells[i][j][grads[x][y]];
                }
            }
        }
    }

    return cells;
}

QVector<double> Model::computeFeatures(const QVector<QVector<Cell> > &cells,
                                       int shift)
{
    QVector<double> descriptor;
    for (int y = 0; y < WIN_HEIGHT_CELL - BLOCK_SIZE + 1; ++ y)
    {
        for (int x = 0; x < WIN_WIDTH_CELL - BLOCK_SIZE + 1; ++x)
        {
            Cell localCell(CELL_SIZE, 0);
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                for (int j = 0; j < BLOCK_SIZE; ++j)
                {
                    for (int k = 0; k < CELL_SIZE; ++k)
                    {
                        localCell[k] += cells[y + i][x + shift +j][k];
                    }
                }
            }
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                for (int j = 0; j < BLOCK_SIZE; ++j)
                {
                    for (int k = 0; k < CELL_SIZE; ++k)
                    {
                        if (localCell[k] == 0)
                        {
                            descriptor.push_back(0);
                            continue;
                        }
                        descriptor.push_back(cells[y + i][x + shift + j][k] / localCell[k]);
                    }
                }
            }
        }
    }

    if (!m_useLinear)
    {
        descriptor = convertFeatures(descriptor);
    }

    return descriptor;
}

QVector<QVector<qreal> > Model::generateFilter()
{
    QVector<QVector<qreal> > filter(3, QVector<qreal>(3, 0));
    filter[0] = QVector<qreal>(3, -1);
    filter[2] = QVector<qreal>(3, 1);
    filter[0][1] = -2;
    filter[2][1] = 2;

    return filter;
}

void Model::sampleNegative()
{
    QStringList img_filters;
    img_filters << "*.png" << "*.jpg" << "*.bmp" << "*.jpeg";
    QDir negDir(m_dirName + "negative/");
    negDir.setNameFilters(img_filters);
    QStringList negative = negDir.entryList(QDir::NoFilter, QDir::Name);

    QTime time = QTime::currentTime();
    qsrand((uint)time.msec());

    qDebug() << "Negative";
    for (int i = 0; i < negative.size(); ++i)
    {
        qDebug() << i << " / " << negative.size();
        QImage image(m_dirName + "negative/" + negative.at(i));

        if (WIN_WIDTH * BACKGROUND_PER_IMG >= image.width())
        {
            QVector<QVector<Cell> > cells = computeCells(image);
            for (int j = 0; j < BACKGROUND_PER_IMG; ++j)
            {
                int x = (qrand() + 1.0) / RAND_MAX * (image.width() / CELL_SIZE
                                                      - WIN_WIDTH_CELL - 1);
                m_features.push_back(computeFeatures(cells, x));
                m_labels.push_back(-1);
            }
        } else {
            for (int j = 0; j < BACKGROUND_PER_IMG; ++j)
            {
                int x = (qrand() + 1.0) / RAND_MAX * (image.width() - WIN_WIDTH_CELL);
                int y = (qrand() + 1.0) / RAND_MAX * (image.height() - WIN_HEIGHT_CELL);
                m_features.push_back(computeFeatures(computeCells(image.copy(
                                                                      x, y,
                                                                      WIN_WIDTH,
                                                                      WIN_HEIGHT))));
                m_labels.push_back(-1);
            }
        }
    }
}

QVector<QImage> Model::getBackgrounds()
{
    QVector<QImage> badExamples;

    QStringList img_filters;
    img_filters << "*.png" << "*.jpg" << "*.bmp" << "*.jpeg";
    QDir negDir(m_dirName + "negative/");
    negDir.setNameFilters(img_filters);
    QStringList negative = negDir.entryList(QDir::NoFilter, QDir::Name);

    qDebug() << "Backgrounds";
    for (int i = 0; i < negative.size(); ++i)
    {
        qDebug() << i << " / " << negative.size();
        QImage image(m_dirName + "negative/" + negative.at(i));
        for (int y = 0; y < image.height() - WIN_HEIGHT; y += BACK_STEP)
        {
            badExamples.push_back(image.copy(0, y, image.width(), WIN_HEIGHT));
        }
        //badExamples.push_back(image);
    }

    return badExamples;
}

QVector<QImage> Model::getPedestrians()
{
    QVector<QImage> goodExamples;

    QStringList img_filters;
    img_filters << "*.png" << "*.jpg" << "*.bmp" << "*.jpeg";
    QDir posDir(m_dirName + "negative/");
    posDir.setNameFilters(img_filters);
    QStringList positive = posDir.entryList(QDir::NoFilter, QDir::Name);

    qDebug() << "Backgrounds";
    for (int i = 0; i < positive.size(); ++i)
    {
        qDebug() << i << " / " << positive.size();
        QImage image(m_dirName + "positive/" + positive.at(i));
        goodExamples.push_back(image);
    }

    return goodExamples;
}

void Model::samplePositive()
{
    QStringList img_filters;
    img_filters << "*.png" << "*.jpg" << "*.bmp" << "*.jpeg";
    QDir posDir(m_dirName + "positive/");
    posDir.setNameFilters(img_filters);
    QStringList positive = posDir.entryList(QDir::NoFilter, QDir::Name);

    qDebug() << "Positive";
    for (int i = 0; i < positive.size(); ++i)
    {
        qDebug() << i << " / " << positive.size();
        QImage image(m_dirName + "positive/" + positive.at(i));
        QVector<QVector<Cell> > cells = computeCells(image);
        QVector<double> descriptor = computeFeatures(cells);
        m_features.push_back(descriptor);
        m_labels.push_back(1);
    }

}

QVector<double> Model::convertFeatures(const QVector<double> &features)
{
        QVector<double> new_vector;
        for (int j = 0; j < features.size(); ++j)
        {
            for (int k = -m_n; k <= m_n; ++k)
            {
                if (features[j] == 0)
                {
                    new_vector.push_back(0);
                    new_vector.push_back(0);
                    continue;
                }
                double lambda = k * m_L;
                double re = qCos(lambda * qLn(features[j]));
                double im = -qSin(lambda * qLn(features[j]));
                double add = qSqrt(features[j] * sech(M_PI * lambda));
                new_vector.push_back(re * add);
                new_vector.push_back(im * add);
            }
        }

    return new_vector;
}
