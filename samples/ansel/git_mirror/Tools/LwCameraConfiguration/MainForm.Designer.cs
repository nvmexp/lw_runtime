namespace LwCameraConfiguration
{
    partial class LwCameraConfigurationForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(LwCameraConfigurationForm));
            this.buttonSave = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.directoryFolderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog();
            this.buttonChooseTempDirectory = new System.Windows.Forms.Button();
            this.buttonSnapshotDirectory = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.textBoxSnapshotDirectory = new System.Windows.Forms.TextBox();
            this.textBoxTempDirectory = new System.Windows.Forms.TextBox();
            this.buttonDefaultTempFolder = new System.Windows.Forms.Button();
            this.buttonDefaultSnapshotsFolder = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.groupBoxOptions = new System.Windows.Forms.GroupBox();
            this.checkBoxSaveCaptureAsPhotoShop = new System.Windows.Forms.CheckBox();
            this.checkBoxAllowBufferOptionsFilter = new System.Windows.Forms.CheckBox();
            this.checkBoxSavePresetWithShot = new System.Windows.Forms.CheckBox();
            this.lblHighresEnhance = new System.Windows.Forms.Label();
            this.sldHighresEnhanceCoeff = new System.Windows.Forms.TrackBar();
            this.labelErrors = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.numerilwpDownCameraSpeed = new System.Windows.Forms.NumerilwpDown();
            this.label10 = new System.Windows.Forms.Label();
            this.buttonResetOptions = new System.Windows.Forms.Button();
            this.label7 = new System.Windows.Forms.Label();
            this.numerilwpDownEyeSeparation = new System.Windows.Forms.NumerilwpDown();
            this.labelLosslessOutput = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.checkBoxLosslessOutput360 = new System.Windows.Forms.CheckBox();
            this.checkBoxLosslessOutputSuperRes = new System.Windows.Forms.CheckBox();
            this.checkBoxRenderDebug = new System.Windows.Forms.CheckBox();
            this.checkBoxKeepShots = new System.Windows.Forms.CheckBox();
            this.checkBoxRemoveCaptureTint = new System.Windows.Forms.CheckBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.numerilwpDownSphereRes = new System.Windows.Forms.NumerilwpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.numerilwpDownHighRes = new System.Windows.Forms.NumerilwpDown();
            this.checkAllowStyleTransferWhileMoving = new System.Windows.Forms.CheckBox();
            this.groupBoxLwCameraStatus = new System.Windows.Forms.GroupBox();
            this.radioButtonDisable = new System.Windows.Forms.RadioButton();
            this.radioButtonEnable = new System.Windows.Forms.RadioButton();
            this.labelLwCameraStatus = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.logLevelComboBox = new System.Windows.Forms.ComboBox();
            this.labelLogLevel = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.btnHotkeyReset = new System.Windows.Forms.Button();
            this.btnHotkeyConfirm = new System.Windows.Forms.Button();
            this.checkAllowNotifications = new System.Windows.Forms.CheckBox();
            this.buttonDefaultStylesFolder = new System.Windows.Forms.Button();
            this.textBoxStylesDirectory = new System.Windows.Forms.TextBox();
            this.buttonStylesDirectory = new System.Windows.Forms.Button();
            this.lblStylesDirectory = new System.Windows.Forms.Label();
            this.checkEnableStyleTransfer = new System.Windows.Forms.CheckBox();
            this.hkeyIlwoke = new LwCameraConfiguration.HotkeyControlLwstom();
            this.groupBoxOptions.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.sldHighresEnhanceCoeff)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownCameraSpeed)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownEyeSeparation)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownSphereRes)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownHighRes)).BeginInit();
            this.groupBoxLwCameraStatus.SuspendLayout();
            this.SuspendLayout();
            // 
            // buttonSave
            // 
            resources.ApplyResources(this.buttonSave, "buttonSave");
            this.buttonSave.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonSave.Name = "buttonSave";
            this.buttonSave.UseVisualStyleBackColor = true;
            this.buttonSave.Click += new System.EventHandler(this.buttonSave_Click);
            // 
            // buttonCancel
            // 
            resources.ApplyResources(this.buttonCancel, "buttonCancel");
            this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
            // 
            // label1
            // 
            resources.ApplyResources(this.label1, "label1");
            this.label1.Name = "label1";
            // 
            // directoryFolderBrowserDialog
            // 
            this.directoryFolderBrowserDialog.RootFolder = System.Environment.SpecialFolder.MyComputer;
            // 
            // buttonChooseTempDirectory
            // 
            resources.ApplyResources(this.buttonChooseTempDirectory, "buttonChooseTempDirectory");
            this.buttonChooseTempDirectory.Name = "buttonChooseTempDirectory";
            this.buttonChooseTempDirectory.UseVisualStyleBackColor = true;
            this.buttonChooseTempDirectory.Click += new System.EventHandler(this.buttonChooseTempDirectory_Click);
            // 
            // buttonSnapshotDirectory
            // 
            resources.ApplyResources(this.buttonSnapshotDirectory, "buttonSnapshotDirectory");
            this.buttonSnapshotDirectory.Name = "buttonSnapshotDirectory";
            this.buttonSnapshotDirectory.UseVisualStyleBackColor = true;
            this.buttonSnapshotDirectory.Click += new System.EventHandler(this.buttonSnapshotDirectory_Click);
            // 
            // label2
            // 
            resources.ApplyResources(this.label2, "label2");
            this.label2.Name = "label2";
            // 
            // textBoxSnapshotDirectory
            // 
            this.textBoxSnapshotDirectory.BackColor = System.Drawing.SystemColors.GradientInactiveCaption;
            this.textBoxSnapshotDirectory.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxSnapshotDirectory.Cursor = System.Windows.Forms.Lwrsors.No;
            resources.ApplyResources(this.textBoxSnapshotDirectory, "textBoxSnapshotDirectory");
            this.textBoxSnapshotDirectory.Name = "textBoxSnapshotDirectory";
            this.textBoxSnapshotDirectory.ReadOnly = true;
            this.textBoxSnapshotDirectory.TabStop = false;
            // 
            // textBoxTempDirectory
            // 
            this.textBoxTempDirectory.BackColor = System.Drawing.SystemColors.GradientInactiveCaption;
            this.textBoxTempDirectory.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxTempDirectory.Cursor = System.Windows.Forms.Lwrsors.No;
            resources.ApplyResources(this.textBoxTempDirectory, "textBoxTempDirectory");
            this.textBoxTempDirectory.Name = "textBoxTempDirectory";
            this.textBoxTempDirectory.ReadOnly = true;
            this.textBoxTempDirectory.TabStop = false;
            // 
            // buttonDefaultTempFolder
            // 
            this.buttonDefaultTempFolder.Image = global::LwCameraConfiguration.Properties.Resources._default;
            resources.ApplyResources(this.buttonDefaultTempFolder, "buttonDefaultTempFolder");
            this.buttonDefaultTempFolder.Name = "buttonDefaultTempFolder";
            this.buttonDefaultTempFolder.UseVisualStyleBackColor = true;
            this.buttonDefaultTempFolder.Click += new System.EventHandler(this.buttonDefaultTempFolder_Click);
            // 
            // buttonDefaultSnapshotsFolder
            // 
            this.buttonDefaultSnapshotsFolder.Image = global::LwCameraConfiguration.Properties.Resources._default;
            resources.ApplyResources(this.buttonDefaultSnapshotsFolder, "buttonDefaultSnapshotsFolder");
            this.buttonDefaultSnapshotsFolder.Name = "buttonDefaultSnapshotsFolder";
            this.buttonDefaultSnapshotsFolder.UseVisualStyleBackColor = true;
            this.buttonDefaultSnapshotsFolder.Click += new System.EventHandler(this.buttonDefaultSnapshotsFolder_Click);
            // 
            // label3
            // 
            resources.ApplyResources(this.label3, "label3");
            this.label3.Name = "label3";
            // 
            // groupBoxOptions
            // 
            this.groupBoxOptions.Controls.Add(this.checkBoxSaveCaptureAsPhotoShop);
            this.groupBoxOptions.Controls.Add(this.checkBoxAllowBufferOptionsFilter);
            this.groupBoxOptions.Controls.Add(this.checkBoxSavePresetWithShot);
            this.groupBoxOptions.Controls.Add(this.lblHighresEnhance);
            this.groupBoxOptions.Controls.Add(this.sldHighresEnhanceCoeff);
            this.groupBoxOptions.Controls.Add(this.labelErrors);
            this.groupBoxOptions.Controls.Add(this.label9);
            this.groupBoxOptions.Controls.Add(this.numerilwpDownCameraSpeed);
            this.groupBoxOptions.Controls.Add(this.label10);
            this.groupBoxOptions.Controls.Add(this.buttonResetOptions);
            this.groupBoxOptions.Controls.Add(this.label7);
            this.groupBoxOptions.Controls.Add(this.numerilwpDownEyeSeparation);
            this.groupBoxOptions.Controls.Add(this.labelLosslessOutput);
            this.groupBoxOptions.Controls.Add(this.label8);
            this.groupBoxOptions.Controls.Add(this.checkBoxLosslessOutput360);
            this.groupBoxOptions.Controls.Add(this.checkBoxLosslessOutputSuperRes);
            this.groupBoxOptions.Controls.Add(this.checkBoxRenderDebug);
            this.groupBoxOptions.Controls.Add(this.checkBoxKeepShots);
            this.groupBoxOptions.Controls.Add(this.checkBoxRemoveCaptureTint);
            this.groupBoxOptions.Controls.Add(this.label6);
            this.groupBoxOptions.Controls.Add(this.label5);
            this.groupBoxOptions.Controls.Add(this.numerilwpDownSphereRes);
            this.groupBoxOptions.Controls.Add(this.label4);
            this.groupBoxOptions.Controls.Add(this.numerilwpDownHighRes);
            this.groupBoxOptions.Controls.Add(this.label3);
            resources.ApplyResources(this.groupBoxOptions, "groupBoxOptions");
            this.groupBoxOptions.Name = "groupBoxOptions";
            this.groupBoxOptions.TabStop = false;
            // 
            // checkBoxSaveCaptureAsPhotoShop
            // 
            resources.ApplyResources(this.checkBoxSaveCaptureAsPhotoShop, "checkBoxSaveCaptureAsPhotoShop");
            this.checkBoxSaveCaptureAsPhotoShop.Name = "checkBoxSaveCaptureAsPhotoShop";
            this.checkBoxSaveCaptureAsPhotoShop.UseVisualStyleBackColor = true;
            // 
            // checkBoxAllowBufferOptionsFilter
            // 
            resources.ApplyResources(this.checkBoxAllowBufferOptionsFilter, "checkBoxAllowBufferOptionsFilter");
            this.checkBoxAllowBufferOptionsFilter.Name = "checkBoxAllowBufferOptionsFilter";
            this.checkBoxAllowBufferOptionsFilter.UseVisualStyleBackColor = true;
            // 
            // checkBoxSavePresetWithShot
            // 
            resources.ApplyResources(this.checkBoxSavePresetWithShot, "checkBoxSavePresetWithShot");
            this.checkBoxSavePresetWithShot.Name = "checkBoxSavePresetWithShot";
            this.checkBoxSavePresetWithShot.UseVisualStyleBackColor = true;
            // 
            // lblHighresEnhance
            // 
            resources.ApplyResources(this.lblHighresEnhance, "lblHighresEnhance");
            this.lblHighresEnhance.Name = "lblHighresEnhance";
            // 
            // sldHighresEnhanceCoeff
            // 
            resources.ApplyResources(this.sldHighresEnhanceCoeff, "sldHighresEnhanceCoeff");
            this.sldHighresEnhanceCoeff.Maximum = 100;
            this.sldHighresEnhanceCoeff.Minimum = 10;
            this.sldHighresEnhanceCoeff.Name = "sldHighresEnhanceCoeff";
            this.sldHighresEnhanceCoeff.TickFrequency = 45;
            this.sldHighresEnhanceCoeff.Value = 35;
            this.sldHighresEnhanceCoeff.Scroll += new System.EventHandler(this.sldHighresEnhanceCoeff_Scroll);
            // 
            // labelErrors
            // 
            resources.ApplyResources(this.labelErrors, "labelErrors");
            this.labelErrors.ForeColor = System.Drawing.Color.Red;
            this.labelErrors.Name = "labelErrors";
            // 
            // label9
            // 
            resources.ApplyResources(this.label9, "label9");
            this.label9.Name = "label9";
            // 
            // numerilwpDownCameraSpeed
            // 
            this.numerilwpDownCameraSpeed.DecimalPlaces = 1;
            this.numerilwpDownCameraSpeed.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            resources.ApplyResources(this.numerilwpDownCameraSpeed, "numerilwpDownCameraSpeed");
            this.numerilwpDownCameraSpeed.Maximum = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.numerilwpDownCameraSpeed.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numerilwpDownCameraSpeed.Name = "numerilwpDownCameraSpeed";
            this.numerilwpDownCameraSpeed.Value = new decimal(new int[] {
            4,
            0,
            0,
            0});
            // 
            // label10
            // 
            resources.ApplyResources(this.label10, "label10");
            this.label10.Name = "label10";
            // 
            // buttonResetOptions
            // 
            resources.ApplyResources(this.buttonResetOptions, "buttonResetOptions");
            this.buttonResetOptions.Image = global::LwCameraConfiguration.Properties.Resources._default;
            this.buttonResetOptions.Name = "buttonResetOptions";
            this.buttonResetOptions.UseVisualStyleBackColor = true;
            this.buttonResetOptions.Click += new System.EventHandler(this.buttonResetOptions_Click);
            // 
            // label7
            // 
            resources.ApplyResources(this.label7, "label7");
            this.label7.Name = "label7";
            // 
            // numerilwpDownEyeSeparation
            // 
            this.numerilwpDownEyeSeparation.DecimalPlaces = 1;
            this.numerilwpDownEyeSeparation.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            resources.ApplyResources(this.numerilwpDownEyeSeparation, "numerilwpDownEyeSeparation");
            this.numerilwpDownEyeSeparation.Maximum = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.numerilwpDownEyeSeparation.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numerilwpDownEyeSeparation.Name = "numerilwpDownEyeSeparation";
            this.numerilwpDownEyeSeparation.Value = new decimal(new int[] {
            63,
            0,
            0,
            65536});
            // 
            // labelLosslessOutput
            // 
            resources.ApplyResources(this.labelLosslessOutput, "labelLosslessOutput");
            this.labelLosslessOutput.Name = "labelLosslessOutput";
            // 
            // label8
            // 
            resources.ApplyResources(this.label8, "label8");
            this.label8.Name = "label8";
            // 
            // checkBoxLosslessOutput360
            // 
            resources.ApplyResources(this.checkBoxLosslessOutput360, "checkBoxLosslessOutput360");
            this.checkBoxLosslessOutput360.Name = "checkBoxLosslessOutput360";
            this.checkBoxLosslessOutput360.UseVisualStyleBackColor = true;
            // 
            // checkBoxLosslessOutputSuperRes
            // 
            resources.ApplyResources(this.checkBoxLosslessOutputSuperRes, "checkBoxLosslessOutputSuperRes");
            this.checkBoxLosslessOutputSuperRes.Name = "checkBoxLosslessOutputSuperRes";
            this.checkBoxLosslessOutputSuperRes.UseVisualStyleBackColor = true;
            // 
            // checkBoxRenderDebug
            // 
            resources.ApplyResources(this.checkBoxRenderDebug, "checkBoxRenderDebug");
            this.checkBoxRenderDebug.Name = "checkBoxRenderDebug";
            this.checkBoxRenderDebug.UseVisualStyleBackColor = true;
            this.checkBoxRenderDebug.CheckedChanged += new System.EventHandler(this.checkBoxRenderDebug_CheckedChanged);
            // 
            // checkBoxKeepShots
            // 
            resources.ApplyResources(this.checkBoxKeepShots, "checkBoxKeepShots");
            this.checkBoxKeepShots.Name = "checkBoxKeepShots";
            this.checkBoxKeepShots.UseVisualStyleBackColor = true;
            // 
            // checkBoxRemoveCaptureTint
            // 
            resources.ApplyResources(this.checkBoxRemoveCaptureTint, "checkBoxRemoveCaptureTint");
            this.checkBoxRemoveCaptureTint.Name = "checkBoxRemoveCaptureTint";
            this.checkBoxRemoveCaptureTint.UseVisualStyleBackColor = true;
            // 
            // label6
            // 
            resources.ApplyResources(this.label6, "label6");
            this.label6.Name = "label6";
            // 
            // label5
            // 
            resources.ApplyResources(this.label5, "label5");
            this.label5.Name = "label5";
            // 
            // numerilwpDownSphereRes
            // 
            resources.ApplyResources(this.numerilwpDownSphereRes, "numerilwpDownSphereRes");
            this.numerilwpDownSphereRes.Maximum = new decimal(new int[] {
            63,
            0,
            0,
            0});
            this.numerilwpDownSphereRes.Minimum = new decimal(new int[] {
            4,
            0,
            0,
            0});
            this.numerilwpDownSphereRes.Name = "numerilwpDownSphereRes";
            this.numerilwpDownSphereRes.Value = new decimal(new int[] {
            8,
            0,
            0,
            0});
            // 
            // label4
            // 
            resources.ApplyResources(this.label4, "label4");
            this.label4.Name = "label4";
            // 
            // numerilwpDownHighRes
            // 
            resources.ApplyResources(this.numerilwpDownHighRes, "numerilwpDownHighRes");
            this.numerilwpDownHighRes.Maximum = new decimal(new int[] {
            128,
            0,
            0,
            0});
            this.numerilwpDownHighRes.Minimum = new decimal(new int[] {
            4,
            0,
            0,
            0});
            this.numerilwpDownHighRes.Name = "numerilwpDownHighRes";
            this.numerilwpDownHighRes.Value = new decimal(new int[] {
            63,
            0,
            0,
            0});
            // 
            // checkAllowStyleTransferWhileMoving
            // 
            resources.ApplyResources(this.checkAllowStyleTransferWhileMoving, "checkAllowStyleTransferWhileMoving");
            this.checkAllowStyleTransferWhileMoving.Name = "checkAllowStyleTransferWhileMoving";
            this.checkAllowStyleTransferWhileMoving.UseVisualStyleBackColor = true;
            // 
            // groupBoxLwCameraStatus
            // 
            this.groupBoxLwCameraStatus.Controls.Add(this.radioButtonDisable);
            this.groupBoxLwCameraStatus.Controls.Add(this.radioButtonEnable);
            this.groupBoxLwCameraStatus.Controls.Add(this.labelLwCameraStatus);
            this.groupBoxLwCameraStatus.Controls.Add(this.label11);
            resources.ApplyResources(this.groupBoxLwCameraStatus, "groupBoxLwCameraStatus");
            this.groupBoxLwCameraStatus.Name = "groupBoxLwCameraStatus";
            this.groupBoxLwCameraStatus.TabStop = false;
            // 
            // radioButtonDisable
            // 
            resources.ApplyResources(this.radioButtonDisable, "radioButtonDisable");
            this.radioButtonDisable.Name = "radioButtonDisable";
            this.radioButtonDisable.TabStop = true;
            this.radioButtonDisable.UseVisualStyleBackColor = true;
            this.radioButtonDisable.CheckedChanged += new System.EventHandler(this.radioButtonDisable_CheckedChanged);
            // 
            // radioButtonEnable
            // 
            resources.ApplyResources(this.radioButtonEnable, "radioButtonEnable");
            this.radioButtonEnable.Name = "radioButtonEnable";
            this.radioButtonEnable.TabStop = true;
            this.radioButtonEnable.UseVisualStyleBackColor = true;
            this.radioButtonEnable.CheckedChanged += new System.EventHandler(this.radioButtonEnable_CheckedChanged);
            // 
            // labelLwCameraStatus
            // 
            resources.ApplyResources(this.labelLwCameraStatus, "labelLwCameraStatus");
            this.labelLwCameraStatus.Name = "labelLwCameraStatus";
            // 
            // label11
            // 
            resources.ApplyResources(this.label11, "label11");
            this.label11.Name = "label11";
            // 
            // logLevelComboBox
            // 
            this.logLevelComboBox.FormattingEnabled = true;
            this.logLevelComboBox.Items.AddRange(new object[] {
            resources.GetString("logLevelComboBox.Items"),
            resources.GetString("logLevelComboBox.Items1"),
            resources.GetString("logLevelComboBox.Items2"),
            resources.GetString("logLevelComboBox.Items3"),
            resources.GetString("logLevelComboBox.Items4"),
            resources.GetString("logLevelComboBox.Items5"),
            resources.GetString("logLevelComboBox.Items6")});
            resources.ApplyResources(this.logLevelComboBox, "logLevelComboBox");
            this.logLevelComboBox.Name = "logLevelComboBox";
            this.logLevelComboBox.SelectedIndexChanged += new System.EventHandler(this.comboBox1_SelectedIndexChanged);
            // 
            // labelLogLevel
            // 
            resources.ApplyResources(this.labelLogLevel, "labelLogLevel");
            this.labelLogLevel.Name = "labelLogLevel";
            // 
            // label12
            // 
            resources.ApplyResources(this.label12, "label12");
            this.label12.Name = "label12";
            // 
            // btnHotkeyReset
            // 
            resources.ApplyResources(this.btnHotkeyReset, "btnHotkeyReset");
            this.btnHotkeyReset.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnHotkeyReset.Name = "btnHotkeyReset";
            this.btnHotkeyReset.UseVisualStyleBackColor = true;
            this.btnHotkeyReset.Click += new System.EventHandler(this.btnHotkeyReset_Click);
            // 
            // btnHotkeyConfirm
            // 
            resources.ApplyResources(this.btnHotkeyConfirm, "btnHotkeyConfirm");
            this.btnHotkeyConfirm.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnHotkeyConfirm.Name = "btnHotkeyConfirm";
            this.btnHotkeyConfirm.UseVisualStyleBackColor = true;
            this.btnHotkeyConfirm.Click += new System.EventHandler(this.btnHotkeyConfirm_Click);
            // 
            // checkAllowNotifications
            // 
            resources.ApplyResources(this.checkAllowNotifications, "checkAllowNotifications");
            this.checkAllowNotifications.Checked = true;
            this.checkAllowNotifications.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkAllowNotifications.Name = "checkAllowNotifications";
            this.checkAllowNotifications.UseVisualStyleBackColor = true;
            // 
            // buttonDefaultStylesFolder
            // 
            this.buttonDefaultStylesFolder.Image = global::LwCameraConfiguration.Properties.Resources._default;
            resources.ApplyResources(this.buttonDefaultStylesFolder, "buttonDefaultStylesFolder");
            this.buttonDefaultStylesFolder.Name = "buttonDefaultStylesFolder";
            this.buttonDefaultStylesFolder.UseVisualStyleBackColor = true;
            this.buttonDefaultStylesFolder.Click += new System.EventHandler(this.buttonDefaultStylesFolder_Click);
            // 
            // textBoxStylesDirectory
            // 
            this.textBoxStylesDirectory.BackColor = System.Drawing.SystemColors.GradientInactiveCaption;
            this.textBoxStylesDirectory.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.textBoxStylesDirectory.Cursor = System.Windows.Forms.Lwrsors.No;
            resources.ApplyResources(this.textBoxStylesDirectory, "textBoxStylesDirectory");
            this.textBoxStylesDirectory.Name = "textBoxStylesDirectory";
            this.textBoxStylesDirectory.ReadOnly = true;
            this.textBoxStylesDirectory.TabStop = false;
            // 
            // buttonStylesDirectory
            // 
            resources.ApplyResources(this.buttonStylesDirectory, "buttonStylesDirectory");
            this.buttonStylesDirectory.Name = "buttonStylesDirectory";
            this.buttonStylesDirectory.UseVisualStyleBackColor = true;
            this.buttonStylesDirectory.Click += new System.EventHandler(this.buttonStylesDirectory_Click);
            // 
            // lblStylesDirectory
            // 
            resources.ApplyResources(this.lblStylesDirectory, "lblStylesDirectory");
            this.lblStylesDirectory.Name = "lblStylesDirectory";
            // 
            // checkEnableStyleTransfer
            // 
            resources.ApplyResources(this.checkEnableStyleTransfer, "checkEnableStyleTransfer");
            this.checkEnableStyleTransfer.Checked = true;
            this.checkEnableStyleTransfer.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkEnableStyleTransfer.Name = "checkEnableStyleTransfer";
            this.checkEnableStyleTransfer.UseVisualStyleBackColor = true;
            // 
            // hkeyIlwoke
            // 
            this.hkeyIlwoke.Hotkey = System.Windows.Forms.Keys.None;
            this.hkeyIlwoke.HotkeyModifiers = System.Windows.Forms.Keys.None;
            resources.ApplyResources(this.hkeyIlwoke, "hkeyIlwoke");
            this.hkeyIlwoke.Name = "hkeyIlwoke";
            this.hkeyIlwoke.Enter += new System.EventHandler(this.hkeyIlwoke_Enter);
            // 
            // LwCameraConfigurationForm
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.checkAllowStyleTransferWhileMoving);
            this.Controls.Add(this.checkEnableStyleTransfer);
            this.Controls.Add(this.lblStylesDirectory);
            this.Controls.Add(this.buttonDefaultStylesFolder);
            this.Controls.Add(this.textBoxStylesDirectory);
            this.Controls.Add(this.buttonStylesDirectory);
            this.Controls.Add(this.checkAllowNotifications);
            this.Controls.Add(this.hkeyIlwoke);
            this.Controls.Add(this.btnHotkeyConfirm);
            this.Controls.Add(this.btnHotkeyReset);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.labelLogLevel);
            this.Controls.Add(this.logLevelComboBox);
            this.Controls.Add(this.groupBoxLwCameraStatus);
            this.Controls.Add(this.groupBoxOptions);
            this.Controls.Add(this.buttonDefaultSnapshotsFolder);
            this.Controls.Add(this.buttonDefaultTempFolder);
            this.Controls.Add(this.textBoxTempDirectory);
            this.Controls.Add(this.textBoxSnapshotDirectory);
            this.Controls.Add(this.buttonSnapshotDirectory);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.buttonChooseTempDirectory);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonSave);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "LwCameraConfigurationForm";
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.Load += new System.EventHandler(this.LwCameraConfigurationForm_Load);
            this.groupBoxOptions.ResumeLayout(false);
            this.groupBoxOptions.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.sldHighresEnhanceCoeff)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownCameraSpeed)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownEyeSeparation)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownSphereRes)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numerilwpDownHighRes)).EndInit();
            this.groupBoxLwCameraStatus.ResumeLayout(false);
            this.groupBoxLwCameraStatus.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button buttonSave;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.FolderBrowserDialog directoryFolderBrowserDialog;
        private System.Windows.Forms.Button buttonChooseTempDirectory;
        private System.Windows.Forms.Button buttonSnapshotDirectory;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBoxSnapshotDirectory;
        private System.Windows.Forms.TextBox textBoxTempDirectory;
        private System.Windows.Forms.Button buttonDefaultTempFolder;
        private System.Windows.Forms.Button buttonDefaultSnapshotsFolder;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.GroupBox groupBoxOptions;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.NumerilwpDown numerilwpDownSphereRes;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.NumerilwpDown numerilwpDownHighRes;
        private System.Windows.Forms.CheckBox checkBoxRemoveCaptureTint;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.NumerilwpDown numerilwpDownCameraSpeed;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Button buttonResetOptions;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.NumerilwpDown numerilwpDownEyeSeparation;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.CheckBox checkBoxRenderDebug;
        private System.Windows.Forms.CheckBox checkBoxKeepShots;
        private System.Windows.Forms.Label labelErrors;
        private System.Windows.Forms.GroupBox groupBoxLwCameraStatus;
        private System.Windows.Forms.RadioButton radioButtonDisable;
        private System.Windows.Forms.RadioButton radioButtonEnable;
        private System.Windows.Forms.Label labelLwCameraStatus;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.CheckBox checkBoxLosslessOutputSuperRes;
        private System.Windows.Forms.Label labelLosslessOutput;
        private System.Windows.Forms.CheckBox checkBoxLosslessOutput360;
        private System.Windows.Forms.ComboBox logLevelComboBox;
        private System.Windows.Forms.Label labelLogLevel;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Button btnHotkeyReset;
        private System.Windows.Forms.Button btnHotkeyConfirm;
        private HotkeyControlLwstom hkeyIlwoke;
        private System.Windows.Forms.CheckBox checkAllowNotifications;
        private System.Windows.Forms.Label lblHighresEnhance;
        private System.Windows.Forms.TrackBar sldHighresEnhanceCoeff;
        private System.Windows.Forms.Button buttonDefaultStylesFolder;
        private System.Windows.Forms.TextBox textBoxStylesDirectory;
        private System.Windows.Forms.Button buttonStylesDirectory;
        private System.Windows.Forms.Label lblStylesDirectory;
        private System.Windows.Forms.CheckBox checkAllowStyleTransferWhileMoving;
        private System.Windows.Forms.CheckBox checkEnableStyleTransfer;
        private System.Windows.Forms.CheckBox checkBoxSavePresetWithShot;
        private System.Windows.Forms.CheckBox checkBoxAllowBufferOptionsFilter;
        private System.Windows.Forms.CheckBox checkBoxSaveCaptureAsPhotoShop;
    }
}

