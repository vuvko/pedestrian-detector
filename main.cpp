#include <view.h>
#include <controller.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    View view;
    Controller controller(&view);
    view.show();

    return app.exec();
}
