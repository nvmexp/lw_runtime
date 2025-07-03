using System;
using System.Collections.Generic;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;


using System.Windows.Forms;
using Microsoft.Win32;

using System.Windows.Input;

namespace LwCameraConfiguration
{
    public partial class LwCameraConfigurationForm : Form
    {
        private const bool isStyleTransferEnabled = true;

        private const Keys defaultToggleHotkeyModifiers = Keys.Alt;
        private const Keys defaultToggleHotkey = Keys.F2;

        private ArrayList ilwalidHotkeys = null;


        private const string lwDrsPathKeyName = @"SOFTWARE\LWPU Corporation\Global\DrsPath";	// HKLM

        private const string lwCameraKeyName = @"SOFTWARE\LWPU Corporation\Ansel";
        private const string lwCameraTempDir = "IntermediateShotsDir";
        private const string lwCameraSnapshotDir = "SnapshotsDir";
        private const string lwCameraUserStylesDir = "UserStylesDir";
        //Options
        private const string lwKeyNameMaxHighRes = "MaxHighRes"; // old "MaxHighResMult";
        private const string lwKeyNameSphereRes = "MaxSphereRes";
        private const string lwKeyNameEyeSep = "StereoEyeSeparation";
        private const string lwKeyNameCameraSpeedMult = "CameraSpeedMult";
        private const string lwKeyNameRemoveTint = "RemoveBlackTint";
        private const string lwKeyNameKeepShots = "KeepIntermediateShots";
        private const string lwKeyNameSavePreset = "SavePresetWithShot";
        private const string lwKeyNameRenderDebug = "RenderDebugInformation";
        private const string lwKeyNameLosslessOutputSuperRes = "LosslessOutput";
        private const string lwKeyNameLosslessOutput360 = "LosslessOutput360";
        private const string lwKeyNameLogFiltering = "LogFiltering";
        private const string lwKeyNameAllowStyleTransferWhileMoving = "AllowStyleTransferWhileMoving";
        private const string lwKeyNameEnableStyleTransfer = "EnableStyleTransfer";

        private const string lwKeyNameAllowNotifications = "AllowNotifications";

        private const string lwKeyNameToggleHotkeyModCtrl = "ToggleHotkeyModCtrl";
        private const string lwKeyNameToggleHotkeyModShift = "ToggleHotkeyModShift";
        private const string lwKeyNameToggleHotkeyModAlt = "ToggleHotkeyModAlt";
        private const string lwKeyNameToggleHotkey = "ToggleHotkey";

        private const string lwKeyNameEnhancedHighresCoeff = "EnhancedHighresCoeff";

        private const string lwKeyNameAllowBufferOptionsFilter = "AllowBufferOptionsFilter";

        private const string lwKeyNameSaveCaptureAsPhotoShop = "SaveCaptureAsPhotoShop";

        private string lwceToolFullpath = "";

        private string intermediateDirectory = "";
        private string snapshotDirectory = "";
        private string userStylesDirectory = "";
        private bool loading = false;

        public void buildIlwalidToggleHotkeysList()
        {
            ilwalidHotkeys = new ArrayList();
            ilwalidHotkeys.Add((int)(Keys.Alt | Keys.Enter));
            ilwalidHotkeys.Add((int)(Keys.Shift | Keys.Insert));
            ilwalidHotkeys.Add((int)(Keys.RWin));
            ilwalidHotkeys.Add((int)(Keys.LWin));
            ilwalidHotkeys.Add((int)(Keys.Alt | Keys.Space));
            ilwalidHotkeys.Add((int)(Keys.Alt | Keys.F4));
            ilwalidHotkeys.Add((int)(Keys.Control | Keys.F4));
            ilwalidHotkeys.Add((int)(Keys.Alt | Keys.Tab));
        }

        public float colwertEnhanceHighresCoeff_ToInternal(int value)
        {
            return value / 100.0f;
        }

        public int colwertEnhanceHighresCoeff_ToDisplay(float value)
        {
            return Colwert.ToInt32(value * 100);
        }

        public LwCameraConfigurationForm()
        {
            InitializeComponent();

            buildIlwalidToggleHotkeysList();
            foreach (var ilwalidHotkey in ilwalidHotkeys)
            {
                hkeyIlwoke.AddIlwalidHotkey((Keys)ilwalidHotkey);
            }

            hkeyIlwoke.SetDefaultHotkey(defaultToggleHotkeyModifiers, defaultToggleHotkey);
        }

        private void buttonCancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void buttonChooseTempDirectory_Click(object sender, EventArgs e)
        {
            directoryFolderBrowserDialog.SelectedPath = intermediateDirectory;
            DialogResult result = directoryFolderBrowserDialog.ShowDialog();
            if(result == DialogResult.OK)
            {
                intermediateDirectory = textBoxTempDirectory.Text = directoryFolderBrowserDialog.SelectedPath + @"\";
            }
            buttonSave.Enabled = (intermediateDirectory != null && snapshotDirectory != null && userStylesDirectory != null);
        }

        private void buttonSnapshotDirectory_Click(object sender, EventArgs e)
        {
            directoryFolderBrowserDialog.SelectedPath = snapshotDirectory;

            DialogResult result = directoryFolderBrowserDialog.ShowDialog();
            if (result == DialogResult.OK)
            {
                snapshotDirectory = textBoxSnapshotDirectory.Text = directoryFolderBrowserDialog.SelectedPath + @"\";
            }
            buttonSave.Enabled = (intermediateDirectory != null && snapshotDirectory != null && userStylesDirectory != null);
        }

        private void buttonSave_Click(object sender, EventArgs e)
        {
            try
            {
                RegistryKey lwCameraKey = Registry.LwrrentUser.CreateSubKey(lwCameraKeyName);

                lwCameraKey.SetValue(lwCameraTempDir, intermediateDirectory);
                lwCameraKey.SetValue(lwCameraSnapshotDir, snapshotDirectory);
                lwCameraKey.SetValue(lwCameraUserStylesDir, userStylesDirectory);
                
                //Options
                lwCameraKey.SetValue(lwKeyNameMaxHighRes, numerilwpDownHighRes.Value.ToString());
                lwCameraKey.SetValue(lwKeyNameSphereRes, numerilwpDownSphereRes.Value.ToString());
                lwCameraKey.SetValue(lwKeyNameEyeSep, numerilwpDownEyeSeparation.Value.ToString());
                lwCameraKey.SetValue(lwKeyNameCameraSpeedMult, numerilwpDownCameraSpeed.Value.ToString());
                //
                lwCameraKey.SetValue(lwKeyNameRemoveTint, checkBoxRemoveCaptureTint.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameKeepShots, checkBoxKeepShots.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameSavePreset, checkBoxSavePresetWithShot.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameAllowBufferOptionsFilter, checkBoxAllowBufferOptionsFilter.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameSaveCaptureAsPhotoShop, checkBoxSaveCaptureAsPhotoShop.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameRenderDebug, checkBoxRenderDebug.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameLosslessOutputSuperRes, checkBoxLosslessOutputSuperRes.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameLosslessOutput360, checkBoxLosslessOutput360.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameAllowStyleTransferWhileMoving, checkAllowStyleTransferWhileMoving.Checked.ToString());
                lwCameraKey.SetValue(lwKeyNameEnableStyleTransfer, checkEnableStyleTransfer.Checked.ToString());
                //
                Keys toggleHotkeyMods = hkeyIlwoke.HotkeyModifiers;
                Keys toggleHotkey = hkeyIlwoke.Hotkey;
                lwCameraKey.SetValue(lwKeyNameToggleHotkeyModCtrl, ((toggleHotkeyMods & Keys.Control) != 0) ? "1" : "0");
                lwCameraKey.SetValue(lwKeyNameToggleHotkeyModShift, ((toggleHotkeyMods & Keys.Shift) != 0) ? "1" : "0");
                lwCameraKey.SetValue(lwKeyNameToggleHotkeyModAlt, ((toggleHotkeyMods & Keys.Alt) != 0) ? "1" : "0");
                int vKey = ((int)toggleHotkey & 0xff);
                lwCameraKey.SetValue(lwKeyNameToggleHotkey, vKey.ToString());

                lwCameraKey.SetValue(lwKeyNameEnhancedHighresCoeff, colwertEnhanceHighresCoeff_ToInternal(sldHighresEnhanceCoeff.Value));

                if (lwCameraKey.GetValue("MaxHighResMult") != null) //remove the old string value that was replaced
                    lwCameraKey.DeleteValue("MaxHighResMult");

                lwCameraKey.SetValue(lwKeyNameAllowNotifications, checkAllowNotifications.Checked.ToString());
                if (logLevelComboBox.SelectedIndex == logLevelComboBox.Items.Count-1)
                {
                    // If logging is disabled we delete the registry value (if it's there)
                    if (lwCameraKey.GetValue(lwKeyNameLogFiltering) != null)
                        lwCameraKey.DeleteValue(lwKeyNameLogFiltering);
                }
                else
                {
                    int logFilteringValue = logLevelComboBox.SelectedIndex - 2;
                    lwCameraKey.SetValue(lwKeyNameLogFiltering, logFilteringValue.ToString());
                }

                lwCameraKey.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString(), "Registry Error", MessageBoxButtons.OK, MessageBoxIcon.Hand);
            }

            Close();
        }

        string GetLwCameraStatus()
        {
            string status = "unknown";
            if (lwceToolFullpath != "")
            {
                Process proc = new Process();
                proc.StartInfo.FileName = lwceToolFullpath;
                proc.StartInfo.Arguments = null;
                proc.StartInfo.UseShellExelwte = false;
                proc.StartInfo.CreateNoWindow = true;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                string output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();
                proc.Close();
                status = ((output[0] == '1') ? "enabled" : "disabled");
            }
            return status;
        }

        bool SetLwCameraStatus(string argument)
        {
            bool status = false;
            if (lwceToolFullpath != "")
            {
                Process proc = new Process();
                proc.StartInfo.FileName = lwceToolFullpath;
                proc.StartInfo.Arguments = argument;
                proc.StartInfo.UseShellExelwte = false;
                proc.StartInfo.CreateNoWindow = true;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                string output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();
                status = (proc.ExitCode == 1);
                proc.Close();
                status = (output[0] == '1');
            }
            return status;
        }

        private void SetDefaultOptions()
        {
            numerilwpDownHighRes.Value = 63;
            numerilwpDownSphereRes.Value = 8;
            numerilwpDownEyeSeparation.Value = (decimal)6.3;
            numerilwpDownCameraSpeed.Value = (decimal)4.0;

            checkBoxRemoveCaptureTint.Checked = false;
            checkBoxKeepShots.Checked = false;
            checkBoxRenderDebug.Checked = false;
            checkBoxLosslessOutputSuperRes.Checked = false;
            checkBoxLosslessOutput360.Checked = false;
            checkBoxSavePresetWithShot.Checked = false;

            logLevelComboBox.SelectedIndex = logLevelComboBox.Items.Count - 1;
        }

        private void SetLwCameraEnableFullpath()
        {
            string path = AppDomain.LwrrentDomain.BaseDirectory;
            if (File.Exists(path + "LwCameraEnable.exe"))
            {
                lwceToolFullpath = path + "LwCameraEnable.exe";
            }
            else
            {
                RegistryKey hklm = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, RegistryView.Registry64);
                RegistryKey lwGlobalKey = null;
                try
                {
                    lwGlobalKey = hklm.OpenSubKey(lwDrsPathKeyName, RegistryKeyPermissionCheck.ReadSubTree);
                }
                catch (Exception)
                {
                    lwGlobalKey = null;
                }

                if (lwGlobalKey != null)
                {
                    path = lwGlobalKey.GetValue("LWDRS_GOLD_PATH").ToString();
                    if (File.Exists(path + @"\LWPU Corporation\Ansel\Tools\LwCameraEnable.exe"))
                    {
                        lwceToolFullpath = path + @"\LWPU Corporation\Ansel\Tools\LwCameraEnable.exe";
                    }
                }
            }
        }

        private void LwCameraConfigurationForm_Load(object sender, EventArgs e)
        {
            // This block controls whether the form contains style transfer settings or not
            if (!isStyleTransferEnabled)
            {
                checkEnableStyleTransfer.Visible = false;
                checkAllowStyleTransferWhileMoving.Visible = false;

                lblStylesDirectory.Visible = false;
                textBoxStylesDirectory.Visible = false;
                buttonDefaultStylesFolder.Visible = false;
                buttonStylesDirectory.Visible = false;

                foreach (Control c in Controls)
                {
                    // Skip Intermediate Directory section
                    if (c.Name == "label1" || c.Name == "textBoxTempDirectory" || c.Name == "buttonDefaultTempFolder" || c.Name == "buttonChooseTempDirectory")
                        continue;

                    // Skip Snapshots Directory section
                    if (c.Name == "label2" || c.Name == "textBoxSnapshotDirectory" || c.Name == "buttonDefaultSnapshotsFolder" || c.Name == "buttonSnapshotDirectory")
                        continue;

                    // Skipping bottom buttons, since they are docked to the form and will move automatically
                    if (c.Name == "buttonSave" || c.Name == "buttonCancel")
                        continue;

                    c.Top -= 70;
                }

                // Decrease form height
                Height -= 70;
            }

            loading = true;

            SetDefaultOptions(); //let's default these first

            // Callwlate LwCameraEnable tool location
            SetLwCameraEnableFullpath();

            labelLwCameraStatus.Text = GetLwCameraStatus();
            if (labelLwCameraStatus.Text == "unknown")
            {
                radioButtonDisable.Checked = radioButtonEnable.Checked = false;
                radioButtonDisable.Enabled = radioButtonEnable.Enabled = false;
            }
            else
            {
                radioButtonEnable.Checked = labelLwCameraStatus.Text == "enabled";
                radioButtonDisable.Checked = labelLwCameraStatus.Text == "disabled";
            }

            RegistryKey lwCameraKey = null;
            try
            {
                lwCameraKey = Registry.LwrrentUser.OpenSubKey(lwCameraKeyName);
            }
            catch (Exception)
            {
                //nothing to do, we just don't have a key
                loading = false;
                return;
            }

            if (lwCameraKey != null)
            {
                object key = lwCameraKey.GetValue(lwCameraTempDir);
                if (key != null)
                {
                    textBoxTempDirectory.Text = key.ToString();
                }

                key = lwCameraKey.GetValue(lwCameraSnapshotDir);
                if (key != null)
                {
                    textBoxSnapshotDirectory.Text = key.ToString();
                }

                key = lwCameraKey.GetValue(lwCameraUserStylesDir);
                if (key != null)
                {
                    textBoxStylesDirectory.Text = key.ToString();
                }


                intermediateDirectory = textBoxTempDirectory.Text;
                snapshotDirectory = textBoxSnapshotDirectory.Text;
                userStylesDirectory = textBoxStylesDirectory.Text;

                bool errors = false;

                //lwKeyNameMaxHighRes = "MaxHighRes";
                key = lwCameraKey.GetValue(lwKeyNameMaxHighRes);
                if (key != null)
                {
                    try
                    {
                        numerilwpDownHighRes.Value = Colwert.ToUInt32(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameSphereRes = "MaxSphereRes";
                key = lwCameraKey.GetValue(lwKeyNameSphereRes);
                if (key != null)
                {
                    try
                    {
                        numerilwpDownSphereRes.Value = Colwert.ToUInt32(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }

                }

                //lwKeyNameEyeSep = "StereoEyeSeparation";
                key = lwCameraKey.GetValue(lwKeyNameEyeSep);
                if (key != null)
                {
                    try
                    {
                        numerilwpDownEyeSeparation.Value = Colwert.ToDecimal(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameCameraSpeedMult = "CameraSpeedMult";
                key = lwCameraKey.GetValue(lwKeyNameCameraSpeedMult);
                if (key != null)
                {
                    try
                    {
                        numerilwpDownCameraSpeed.Value = Colwert.ToDecimal(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameRemoveTint = "RemoveBlackTint";
                key = lwCameraKey.GetValue(lwKeyNameRemoveTint);
                if (key != null)
                {
                    try
                    {
                        checkBoxRemoveCaptureTint.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameAllowStyleTransferWhileMoving = ""AllowStyleTransferWhileMoving"";
                key = lwCameraKey.GetValue(lwKeyNameAllowStyleTransferWhileMoving);
                if (key != null)
                {
                    try
                    {
                        checkAllowStyleTransferWhileMoving.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameEnableStyleTransfer = ""EnableStyleTransfer"";
                key = lwCameraKey.GetValue(lwKeyNameEnableStyleTransfer);
                if (key != null)
                {
                    try
                    {
                        checkEnableStyleTransfer.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameKeepShots = "KeepIntermediateShots";
                key = lwCameraKey.GetValue(lwKeyNameKeepShots);
                if (key != null)
                {
                    try
                    {
                        checkBoxKeepShots.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameSavePreset = "SavePresetWithShot";
                key = lwCameraKey.GetValue(lwKeyNameSavePreset);
                if (key != null)
                {
                    try
                    {
                        checkBoxSavePresetWithShot.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameAllowBufferOptionsFilter = "AllowBufferOptionsFilter";
                key = lwCameraKey.GetValue(lwKeyNameAllowBufferOptionsFilter);
                if (key != null)
                {
                    try
                    {
                        checkBoxAllowBufferOptionsFilter.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameSaveCaptureAsPhotoShop = "SaveCaptureAsPhotoShop";
                key = lwCameraKey.GetValue(lwKeyNameSaveCaptureAsPhotoShop);
                if (key != null)
                {
                    try
                    {
                        checkBoxSaveCaptureAsPhotoShop.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameRenderDebug = "RenderDebugInformation";
                key = lwCameraKey.GetValue(lwKeyNameRenderDebug);
                if (key != null)
                {
                    try
                    {
                        checkBoxRenderDebug.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameLosslessOutputSuperRes = "LosslessOutput";
                key = lwCameraKey.GetValue(lwKeyNameLosslessOutputSuperRes);
                if (key != null)
                {
                    try
                    {
                        checkBoxLosslessOutputSuperRes.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                //lwKeyNameLosslessOutput360 = "LosslessOutput360";
                key = lwCameraKey.GetValue(lwKeyNameLosslessOutput360);
                if (key != null)
                {
                    try
                    {
                        checkBoxLosslessOutput360.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                key = lwCameraKey.GetValue(lwKeyNameLogFiltering);
                if (key != null)
                {
                    try
                    {
                        int index = Colwert.ToInt32(key.ToString());
                        index += 2;
                        int maxIndex = logLevelComboBox.Items.Count - 1;
                        if (index > maxIndex)
                            index = maxIndex;
                        logLevelComboBox.SelectedIndex = index;
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                key = lwCameraKey.GetValue(lwKeyNameAllowNotifications);
                if (key != null)
                {
                    try
                    {
                        checkAllowNotifications.Checked = Colwert.ToBoolean(key.ToString());
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                Keys toggleHotkey = Keys.None;
                Keys toggleHotkeyMods = Keys.None;

                // Ctrl
                key = lwCameraKey.GetValue(lwKeyNameToggleHotkeyModCtrl);
                if (key != null)
                {
                    try
                    {
                        if (Colwert.ToInt32(key.ToString()) != 0)
                            toggleHotkeyMods = toggleHotkeyMods | Keys.Control;
                    }
                    catch (Exception) { errors = true; }
                }
                // Shift
                key = lwCameraKey.GetValue(lwKeyNameToggleHotkeyModShift);
                if (key != null)
                {
                    try
                    {
                        if (Colwert.ToInt32(key.ToString()) != 0)
                            toggleHotkeyMods = toggleHotkeyMods | Keys.Shift;
                    }
                    catch (Exception) { errors = true; }
                }
                // Alt
                key = lwCameraKey.GetValue(lwKeyNameToggleHotkeyModAlt);
                if (key != null)
                {
                    try
                    {
                        if (Colwert.ToInt32(key.ToString()) != 0)
                            toggleHotkeyMods = toggleHotkeyMods | Keys.Alt;
                    }
                    catch (Exception) { errors = true; }
                }

                // Key
                key = lwCameraKey.GetValue(lwKeyNameToggleHotkey);
                if (key != null)
                {
                    try { toggleHotkey = (Keys)Colwert.ToInt32(key.ToString()); }
                    catch (Exception) { errors = true; }
                }

                hkeyIlwoke.SetHotkey(toggleHotkeyMods, toggleHotkey);

                key = lwCameraKey.GetValue(lwKeyNameEnhancedHighresCoeff);
                if (key != null)
                {
                    try
                    {
                        sldHighresEnhanceCoeff.Value = colwertEnhanceHighresCoeff_ToDisplay(Colwert.ToSingle(key.ToString()));
                    }
                    catch (Exception)
                    {
                        errors = true;
                    }
                }

                lwCameraKey.Close();

                labelErrors.Visible = errors;
            }
            loading = false;

            formatLabel_lblHighresEnhance();
        }

        private void buttonDefaultTempFolder_Click(object sender, EventArgs e)
        {
            intermediateDirectory = textBoxTempDirectory.Text = System.Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData).ToString() + @"\Temp\";
        }

        private void buttonDefaultSnapshotsFolder_Click(object sender, EventArgs e)
        {
            snapshotDirectory = textBoxSnapshotDirectory.Text = System.Environment.GetFolderPath(Environment.SpecialFolder.MyPictures).ToString() + @"\";
        }

        private void buttonResetOptions_Click(object sender, EventArgs e)
        {
            SetDefaultOptions();
        }

        private void radioButtonEnable_CheckedChanged(object sender, EventArgs e)
        {
            if(!loading)
                labelLwCameraStatus.Text = SetLwCameraStatus("enable") ? "enabled" : "unknown";
        }

        private void radioButtonDisable_CheckedChanged(object sender, EventArgs e)
        {
            if (!loading)
                labelLwCameraStatus.Text = SetLwCameraStatus("disable") ? "disabled" : "unknown";
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void btnHotkeyReset_Click(object sender, EventArgs e)
        {
            hkeyIlwoke.SetHotkey(defaultToggleHotkeyModifiers, defaultToggleHotkey);
        }

        private void btnHotkeyConfirm_Click(object sender, EventArgs e)
        {
            hkeyIlwoke.ConfirmHotkey();
        }

        private void hkeyIlwoke_Enter(object sender, EventArgs e)
        {
            hkeyIlwoke.OnFolws();
        }

        private void formatLabel_lblHighresEnhance()
        {
            lblHighresEnhance.Text = "High Resolution Capture Enhancement Intensity: " + sldHighresEnhanceCoeff.Value.ToString() + @"%";
        }

        private void sldHighresEnhanceCoeff_Scroll(object sender, EventArgs e)
        {
            formatLabel_lblHighresEnhance();
        }

        private void buttonStylesDirectory_Click(object sender, EventArgs e)
        {
            directoryFolderBrowserDialog.SelectedPath = userStylesDirectory;

            DialogResult result = directoryFolderBrowserDialog.ShowDialog();
            if (result == DialogResult.OK)
            {
                userStylesDirectory = textBoxStylesDirectory.Text = directoryFolderBrowserDialog.SelectedPath + @"\";
            }
            buttonSave.Enabled = (intermediateDirectory != null && snapshotDirectory != null && userStylesDirectory != null);
        }

        private void buttonDefaultStylesFolder_Click(object sender, EventArgs e)
        {
            userStylesDirectory = textBoxStylesDirectory.Text = System.Environment.GetFolderPath(Environment.SpecialFolder.MyPictures).ToString() + @"\Ansel\Styles\";
        }

        private void checkBoxRenderDebug_CheckedChanged(object sender, EventArgs e)
        {

        }
    }
}
