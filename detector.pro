QT       += gui

TEMPLATE = app

SOURCES += \
    model.cpp \
    controller.cpp \
    main.cpp \
    liblinear-1.8/tron.cpp \
    liblinear-1.8/linear.cpp \
    liblinear-1.8/blas/dscal.c \
    liblinear-1.8/blas/dnrm2.c \
    liblinear-1.8/blas/ddot.c \
    liblinear-1.8/blas/daxpy.c \
    view.cpp \

HEADERS += \
    model.h \
    controller.h \
    liblinear-1.8/tron.h \
    liblinear-1.8/linear.h \
    liblinear-1.8/blas/blasp.h \
    liblinear-1.8/blas/blas.h \
    iview.h \
    view.h

FORMS += \
    view.ui


































