// -*- C++ -*-

#include "qtlualistener.h"

#include <QEvent>
#include <QFolwsEvent>
#include <QKeyEvent>
#include <QMetaEnum>
#include <QMetaObject>
#include <QMouseEvent>
#include <QObject>
#include <QPaintEvent>
#include <QResizeEvent>


#ifndef Q_MOC_RUN
static QMetaEnum 
f_enumerator(const char *s)
{
  struct QFakeObject : public QObject {
    static const QMetaObject* qt() { return &staticQtMetaObject; }
  };
  const QMetaObject *mo = QFakeObject::qt();
  int index = (mo) ? mo->indexOfEnumerator(s) : -1;
  if (index >= 0)
    return mo->enumerator(index);
  return QMetaEnum();
}
#endif



QtLuaListener::QtLuaListener(QWidget *w)
  : QObject(w), w(w)
{
  w->installEventFilter(this);
}


bool 
QtLuaListener::eventFilter(QObject *object, QEvent *event)
{
  switch(event->type())
    {
    case QEvent::Close:
      {
        emit sigClose();
        break;
      }
    case QEvent::Resize: 
      { 
        QSize s = static_cast<QResizeEvent*>(event)->size();
        emit sigResize(s.width(), s.height());
        break;
      }
    case QEvent::KeyPress:
    case QEvent::KeyRelease:
      {
        QKeyEvent *e = static_cast<QKeyEvent*>(event);
        QMetaEnum ek = f_enumerator("Key");
        QMetaEnum em = f_enumerator("KeyboardModifiers");
        QByteArray k = ek.valueToKey(e->key());
        QByteArray m = em.valueToKeys(e->modifiers());
        if (event->type() == QEvent::KeyPress)
          emit sigKeyPress(e->text(), k, m);
        else
          emit sigKeyRelease(e->text(), k, m);          
        break;
      }
    case QEvent::MouseButtonPress:
    case QEvent::MouseButtonRelease:
    case QEvent::MouseButtonDblClick:
    case QEvent::MouseMove:
      {
        QMouseEvent *e = static_cast<QMouseEvent*>(event);
        QMetaEnum em = f_enumerator("KeyboardModifiers");
        QMetaEnum eb = f_enumerator("MouseButtons");
        QByteArray b = eb.valueToKey(e->button());
        QByteArray m = em.valueToKeys(e->modifiers());
        QByteArray s = eb.valueToKeys(e->buttons());
        if (event->type() == QEvent::MouseButtonPress)
          emit sigMousePress(e->x(), e->y(), b, m, s);
        else if (event->type() == QEvent::MouseButtonRelease)
          emit sigMouseRelease(e->x(), e->y(), b, m, s);
        else if (event->type() == QEvent::MouseButtonDblClick)
          emit sigMouseDoubleClick(e->x(), e->y(), b, m, s);
        else
          emit sigMouseMove(e->x(), e->y(), m, s);
        break;
      }
    case QEvent::Enter:
    case QEvent::Leave:
      {
        emit sigEnter(event->type() == QEvent::Enter);
        break;
      }
    case QEvent::FolwsIn:
    case QEvent::FolwsOut:
      {
        emit sigFolws(event->type() == QEvent::FolwsIn);
        break;
      }
    case QEvent::Show:
    case QEvent::Hide:
      {
        emit sigShow(event->type() == QEvent::Show);
        break;
      }
    case QEvent::Paint:
      {
        emit sigPaint();
        break;
      }
    default:
      break;
    }
  return false;
}




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

