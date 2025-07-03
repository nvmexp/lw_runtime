using System;
using System.Collections.Generic;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

//using System.Diagnostics;

namespace LwCameraConfiguration
{
    public class HotkeyControlLwstom : TextBox
    {
        private bool areAllKeysReleased = true;

        private Keys defaultHotkey = Keys.None;
        private Keys defaultHotkeyModifiers = Keys.None;

        private Keys lastValidHotkey = Keys.None;
        private Keys lastValidHotkeyModifiers = Keys.None;

        private Keys _hotkey = Keys.None;
        private Keys _modifiers = Keys.None;

        private ArrayList ilwalidHotkeys = null;

        // Overriding default TextBox behavior
        /////////////////////////////////////////////////////////
        private ContextMenu emptyContextMenu = new ContextMenu();

        // Disallow context menu (force to empty)
        public override ContextMenu ContextMenu
        {
            get { return emptyContextMenu; }
            set { base.ContextMenu = emptyContextMenu; }
        }

        // Disallow setting the box multiline
        public override bool Multiline
        {
            get { return base.Multiline; }
            set { base.Multiline = false; }
        }

        public HotkeyControlLwstom()
        {
            // Trigger overrides
            this.ContextMenu = emptyContextMenu;
            this.Multiline = false;

            this.Text = "None";

            this.KeyPress += new KeyPressEventHandler(HotkeyControlLwstom_KeyPress);
            this.KeyUp += new KeyEventHandler(HotkeyControlLwstom_KeyUp);
            this.KeyDown += new KeyEventHandler(HotkeyControlLwstom_KeyDown);

            // Build invalid controls list
            ilwalidHotkeys = new ArrayList();
        }

        public bool IsHotkeyComboValid()
        {
            if ((this._hotkey == Keys.None) || this.ilwalidHotkeys.Contains((int)(this._hotkey | this._modifiers)))
            {
                return false;
            }
            return true;
        }

        public void SetHotkey(Keys modifier, Keys hotkey)
        {
            this.Hotkey = hotkey;
            this.HotkeyModifiers = modifier;
            this.ConfirmHotkey();
        }
        public void SetDefaultHotkey(Keys modifier, Keys hotkey)
        {
            this.defaultHotkey = hotkey;
            this.defaultHotkeyModifiers = modifier;
        }

        public void ConfirmHotkey()
        {
            if (IsHotkeyComboValid())
            {
                this.lastValidHotkey = this.Hotkey;
                this.lastValidHotkeyModifiers = this.HotkeyModifiers;
            }
            else
            {
                ResetHotkey();
            }
        }

        public void RestoreDefaultHotkey()
        {
            this.Hotkey = defaultHotkey;
            this.HotkeyModifiers = defaultHotkeyModifiers;
            this.lastValidHotkey = defaultHotkey;
            this.lastValidHotkeyModifiers = defaultHotkeyModifiers;
        }

        public void ResetHotkey()
        {
            areAllKeysReleased = true;

            this._hotkey = lastValidHotkey;
            this._modifiers = lastValidHotkeyModifiers;

            if (!IsHotkeyComboValid())
            {
                RestoreDefaultHotkey();
            }

            UpdateText();
        }

        public void AddIlwalidHotkey(Keys keycombo)
        {
            ilwalidHotkeys.Add((int)keycombo);
        }

        public void OnFolws()
        {
            areAllKeysReleased = true;
        }

        void HotkeyControlLwstom_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Back || e.KeyCode == Keys.Delete)
            {
                ResetHotkey();
                return;
            }
            else
            {
                this._modifiers = e.Modifiers;

                bool isNonModifierKeyPressed = false;
                // KeyEventArgs.KeyCode will have modifier keys too if they ilwoked the event
                if (e.KeyCode != Keys.ShiftKey && e.KeyCode != Keys.ControlKey && e.KeyCode != Keys.Menu)
                {
                    isNonModifierKeyPressed = true;
                    this._hotkey = e.KeyCode;
                }
                else if (areAllKeysReleased)
                {
                    this._hotkey = Keys.None;
                }

                if (isNonModifierKeyPressed)
                {
                    ConfirmHotkey();
                }

                areAllKeysReleased = false;
                UpdateText();
            }
        }

        void HotkeyControlLwstom_KeyUp(object sender, KeyEventArgs e)
        {
            if (this.Folwsed && this._hotkey == e.KeyCode)
            {
                this._hotkey = Keys.None;
                UpdateText();
            }

            if (this._hotkey == Keys.None && Control.ModifierKeys == Keys.None)
            {
                ResetHotkey();
                return;
            }
        }

        void HotkeyControlLwstom_KeyPress(object sender, KeyPressEventArgs e)
        {
            e.Handled = true;
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (this.ilwalidHotkeys.Contains((int)keyData))
                return true;

            return base.ProcessCmdKey(ref msg, keyData);
        }

        public Keys Hotkey
        {
            get
            {
                return this._hotkey;
            }
            set
            {
                this._hotkey = value;
                UpdateText();
            }
        }

        public Keys HotkeyModifiers
        {
            get
            {
                return this._modifiers;
            }
            set
            {
                this._modifiers = value;
                UpdateText();
            }
        }

        private void UpdateText()
        {
            // No hotkey set
            if (this._hotkey == Keys.None && this._modifiers == Keys.None)
            {
                this.Text = "None";
                return;
            }

            string hotkeyText = "";
            if (this._hotkey != Keys.None)
                hotkeyText = this._hotkey.ToString();

            string modifierText = "";
            if ((this._modifiers & Keys.Control) != 0)
                modifierText += "Ctrl + ";
            if ((this._modifiers & Keys.Shift) != 0)
                modifierText += "Shift + ";
            if ((this._modifiers & Keys.Alt) != 0)
                modifierText += "Alt + ";

            this.Text = modifierText + hotkeyText;
        }
    }
}
