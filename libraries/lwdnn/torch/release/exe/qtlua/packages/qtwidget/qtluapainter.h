// -*- C++ -*-

#ifndef QTLUAPAINTER_H
#define QTLUAPAINTER_H

#include "lua.h"
#include "lauxlib.h"
#include "qtluaengine.h"
#include "qtluautils.h"

#include "qtwidget.h"

#include <QBrush>
#include <QByteArray>
#include <QFlags>
#include <QMetaType>
#include <QImage>
#include <QObject>
#include <QPaintDevice>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPoint>
#include <QPrinter>
#include <QRegion>
#include <QTransform>
#include <QVariant>
#include <QWidget>

class QEvent;
class QCloseEvent;
class QFolwsEvent;
class QPaintEvent;
class QResizeEvent;
class QKeyEvent;
class QMouseEvent;
class QMainWindow;
class QtLuaPrinter;

Q_DECLARE_METATYPE(QGradient)
Q_DECLARE_METATYPE(QPainterPath)
Q_DECLARE_METATYPE(QPolygon)
Q_DECLARE_METATYPE(QPolygonF)
Q_DECLARE_METATYPE(QPainter*)
Q_DECLARE_METATYPE(QPrinter*)
Q_DECLARE_METATYPE(QPaintDevice*)

class QTWIDGET_API QtLuaPainter : public QObject
{
  Q_OBJECT
  Q_PROPERTY(QPen pen READ lwrrentpen WRITE setpen)
  Q_PROPERTY(QBrush brush READ lwrrentbrush WRITE setbrush)
  Q_PROPERTY(QPointF point READ lwrrentpoint WRITE setpoint)
  Q_PROPERTY(QPainterPath path READ lwrrentpath WRITE setpath)
  Q_PROPERTY(QPainterPath clippath READ lwrrentclip WRITE setclip)
  Q_PROPERTY(QFont font READ lwrrentfont WRITE setfont)
  Q_PROPERTY(QTransform matrix READ lwrrentmatrix WRITE setmatrix)
    // special
  Q_PROPERTY(QBrush background READ lwrrentbackground WRITE setbackground)
  Q_PROPERTY(CompositionMode compositionmode READ lwrrentmode WRITE setmode)
  Q_PROPERTY(RenderHints renderhints READ lwrrenthints WRITE sethints)
  Q_PROPERTY(AngleUnit angleUnit READ lwrrentangleunit WRITE setangleunit)
  Q_PROPERTY(QString styleSheet READ lwrrentstylesheet WRITE setstylesheet)
  Q_PROPERTY(int width READ width)
  Q_PROPERTY(int height READ height)
  Q_PROPERTY(int depth READ depth)
  Q_ENUMS(CompositionMode AngleUnit)
  Q_FLAGS(RenderHints TextFlags)

public:
  ~QtLuaPainter();
  QtLuaPainter();
  QtLuaPainter(QImage image);
  QtLuaPainter(QPixmap pixmap);
  QtLuaPainter(int w, int h, bool monochrome=false);
  QtLuaPainter(QString fileName, const char *format = 0);
  QtLuaPainter(QWidget *widget, bool buffered=true);
  QtLuaPainter(QObject *object);

  Q_ILWOKABLE QImage image() const;
  Q_ILWOKABLE QPixmap pixmap() const;
  Q_ILWOKABLE QWidget *widget() const;  
  Q_ILWOKABLE QObject *object() const;
  Q_ILWOKABLE QPaintDevice *device() const;
  Q_ILWOKABLE QPrinter *printer() const;
  Q_ILWOKABLE QPainter *painter() const;
  Q_ILWOKABLE QRect rect() const;
  Q_ILWOKABLE QSize size() const;
  Q_ILWOKABLE void close();
  int width() const { return size().width(); }
  int height() const { return size().height(); }
  int depth() const;

  enum AngleUnit { Degrees, Radians };

  // copy qpainter enums for moc!
  enum CompositionMode {
    SourceOver = QPainter::CompositionMode_SourceOver,
    DestinationOver = QPainter::CompositionMode_DestinationOver,
    Clear = QPainter::CompositionMode_Clear,
    Source = QPainter::CompositionMode_Source,
    Destination = QPainter::CompositionMode_Destination,
    SourceIn = QPainter::CompositionMode_SourceIn,
    DestinationIn = QPainter::CompositionMode_DestinationIn,
    SourceOut = QPainter::CompositionMode_SourceOut,
    DestinationOut = QPainter::CompositionMode_DestinationOut,
    SourceAtop = QPainter::CompositionMode_SourceAtop,
    DestinationAtop = QPainter::CompositionMode_DestinationAtop,
    Xor = QPainter::CompositionMode_Xor,
    Plus = QPainter::CompositionMode_Plus,
    Multiply = QPainter::CompositionMode_Multiply,
    Screen = QPainter::CompositionMode_Screen,
    Overlay = QPainter::CompositionMode_Overlay,
    Darken = QPainter::CompositionMode_Darken,
    Lighten = QPainter::CompositionMode_Lighten,
    ColorDodge = QPainter::CompositionMode_ColorDodge,
    ColorBurn = QPainter::CompositionMode_ColorBurn,
    HardLight = QPainter::CompositionMode_HardLight,
    SoftLight = QPainter::CompositionMode_SoftLight,
    Difference = QPainter::CompositionMode_Difference,
    Exclusion = QPainter::CompositionMode_Exclusion
  };
  enum RenderHint {
    Antialiasing = QPainter::Antialiasing,
    TextAntialiasing = QPainter::TextAntialiasing,
    SmoothPixmapTransform = QPainter::SmoothPixmapTransform,
    HighQualityAntialiasing = QPainter::HighQualityAntialiasing,
  };
  enum TextFlag {
    AlignLeft = Qt::AlignLeft,
    AlignRight = Qt::AlignRight,
    AlignHCenter = Qt::AlignHCenter,
    AlignJustify = Qt::AlignJustify,
    AlignTop = Qt::AlignTop,
    AlignBottom = Qt::AlignBottom,
    AliglwCenter = Qt::AliglwCenter,
    AlignCenter = Qt::AlignCenter,
    TextSingleLine  =Qt::TextSingleLine,
    TextExpandTabs = Qt::TextExpandTabs,
    TextShowMnemonic = Qt::TextShowMnemonic,
    TextWordWrap = Qt::TextWordWrap,
    TextRich = (Qt::TextWordWrap|Qt::TextSingleLine), // magic
    RichText = TextRich                               // alias
  };
  Q_DECLARE_FLAGS(RenderHints,RenderHint);
  Q_DECLARE_FLAGS(TextFlags,TextFlag);

public slots:
  virtual void showpage();
  void refresh();
  
public:
  // buffering
  void gbegin();
  void gend(bool ilwalidate=false);
  // state
  QPen lwrrentpen() const;
  QBrush lwrrentbrush() const;
  QPointF lwrrentpoint() const;
  QPainterPath lwrrentpath() const;
  QPainterPath lwrrentclip() const;
  QFont lwrrentfont() const;
  QTransform lwrrentmatrix() const;
  QBrush lwrrentbackground() const;
  CompositionMode lwrrentmode() const;
  RenderHints lwrrenthints() const;
  AngleUnit lwrrentangleunit() const;
  QString lwrrentstylesheet() const;
  void setpen(QPen pen);
  void setbrush(QBrush brush);
  void setpoint(QPointF p);
  void setpath(QPainterPath p);
  void setclip(QPainterPath p);
  void setfont(QFont f);
  void setmatrix(QTransform m);
  void setbackground(QBrush brush);
  void setmode(CompositionMode m);
  void sethints(RenderHints h);
  void setangleunit(AngleUnit u);
  void setstylesheet(QString s);
  // postscript rendering
  void initclip();
  void initmatrix();
  void initgraphics();
  void scale(qreal x, qreal y);
  void rotate(qreal x);
  void translate(qreal x, qreal y);
  void concat(QTransform m);
  void gsave();
  void grestore();
  void newpath();
  void moveto(qreal x, qreal y);
  void lineto(qreal x, qreal y);
  void lwrveto(qreal x1, qreal y1, qreal x2, qreal y2, qreal x3, qreal y3);
  void arc(qreal x, qreal y, qreal r, qreal a1, qreal a2);
  void arcn(qreal x, qreal y, qreal r, qreal a1, qreal a2);
  void arcto(qreal x1, qreal y1, qreal x2, qreal y2, qreal r);
  void rmoveto(qreal x, qreal y);
  void rlineto(qreal x, qreal y);
  void rlwrveto(qreal x1, qreal y1, qreal x2, qreal y2, qreal x3, qreal y3);
  void charpath(QString text);
  void closepath();
  void stroke(bool resetpath=true);
  void fill(bool resetpath=true);
  void eofill(bool resetpath=true);
  void clip(bool resetpath=false);
  void eoclip(bool resetpath=false);
  void show(QString text);
  qreal stringwidth(QString text, qreal *pdx=0, qreal *pdy=0);
  // additional useful functions
  void rectangle(qreal x, qreal y, qreal w, qreal h); // non ps
  void image(QRectF drect, QImage i, QRectF srect);
  void image(QRectF drect, QPixmap p, QRectF srect);
  void image(QRectF drect, QtLuaPainter *p, QRectF srect);
  void show(QString text, qreal x, qreal y, qreal w, qreal h, int flags=0);
  QRectF stringrect(QString text);
  QRectF stringrect(QString text, qreal x, qreal y, qreal w, qreal h, int f=0);
  
public:
  struct Private;
  struct Locker;
  struct State;
protected:
  Private *d;
};



#endif


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*" "qreal")
   End:
   ------------------------------------------------------------- */


