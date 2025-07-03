#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "TearTestApp.h"

extern bool fullscreen;
extern int vsync;
extern char *resString;
extern TearTestApp tt;
extern char *apiString;

char settingsResString[128];

namespace ms {

    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Collections;
    using namespace System::Windows::Forms;
    using namespace System::Data;
    using namespace System::Drawing;
    using namespace System::Runtime::InteropServices;

    /// <summary>
    /// Summary for SettingsDialog
    /// </summary>
    public ref class SettingsDialog : public System::Windows::Forms::Form
    {
    public:
        SettingsDialog(void)
        {
            InitializeComponent();
            //
            //TODO: Add the constructor code here
            //
            comboBoxAPI->SelectedIndex = 0;
            set< pair<int, int> > supRes;
            DEVMODE dm = { 0 };
            dm.dmSize = sizeof(dm);
            for (int i = 0; EnumDisplaySettings(NULL, i, &dm) != 0; i++)
            {
                supRes.insert(pair<int, int>(dm.dmPelsWidth, dm.dmPelsHeight));
            }

            for (auto&p : supRes)
            {
                listBoxRes->Items->Add(String::Format("{0}x{1}", p.first, p.second));
            }
        }

    protected:
        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        ~SettingsDialog()
        {
            if (components)
            {
                delete components;
            }
        }

    protected:
    private: System::Windows::Forms::ListBox^  listBoxRes;
    private: System::Windows::Forms::CheckBox^  checkBoxFS;
    private: System::Windows::Forms::Button^  buttonOK;
    private: System::Windows::Forms::CheckBox^  checkBoxVSync;
    private: System::Windows::Forms::Button^  buttonBgnd;
    private: System::Windows::Forms::GroupBox^  groupBox1;
    private: System::Windows::Forms::GroupBox^  groupBox2;
    private: System::Windows::Forms::CheckBox^  checkBoxGradient;
    private: System::Windows::Forms::GroupBox^  groupBox3;
    private: System::Windows::Forms::ComboBox^  comboBoxAPI;

    private:
        /// <summary>
        /// Required designer variable.
        /// </summary>
        System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        void InitializeComponent(void)
        {
            System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(SettingsDialog::typeid));
            this->listBoxRes = (gcnew System::Windows::Forms::ListBox());
            this->checkBoxFS = (gcnew System::Windows::Forms::CheckBox());
            this->buttonOK = (gcnew System::Windows::Forms::Button());
            this->checkBoxVSync = (gcnew System::Windows::Forms::CheckBox());
            this->buttonBgnd = (gcnew System::Windows::Forms::Button());
            this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
            this->checkBoxGradient = (gcnew System::Windows::Forms::CheckBox());
            this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
            this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
            this->comboBoxAPI = (gcnew System::Windows::Forms::ComboBox());
            this->groupBox1->SuspendLayout();
            this->groupBox2->SuspendLayout();
            this->groupBox3->SuspendLayout();
            this->SuspendLayout();
            //
            // listBoxRes
            //
            this->listBoxRes->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->listBoxRes->FormattingEnabled = true;
            this->listBoxRes->Location = System::Drawing::Point(6, 26);
            this->listBoxRes->Name = L"listBoxRes";
            this->listBoxRes->Size = System::Drawing::Size(228, 342);
            this->listBoxRes->TabIndex = 1;
            //
            // checkBoxFS
            //
            this->checkBoxFS->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->checkBoxFS->AutoSize = true;
            this->checkBoxFS->Checked = true;
            this->checkBoxFS->CheckState = System::Windows::Forms::CheckState::Checked;
            this->checkBoxFS->Location = System::Drawing::Point(8, 375);
            this->checkBoxFS->Name = L"checkBoxFS";
            this->checkBoxFS->Size = System::Drawing::Size(74, 17);
            this->checkBoxFS->TabIndex = 2;
            this->checkBoxFS->Text = L"Fullscreen";
            this->checkBoxFS->UseVisualStyleBackColor = true;
            //
            // buttonOK
            //
            this->buttonOK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->buttonOK->Location = System::Drawing::Point(12, 544);
            this->buttonOK->Name = L"buttonOK";
            this->buttonOK->Size = System::Drawing::Size(240, 23);
            this->buttonOK->TabIndex = 7;
            this->buttonOK->Text = L"OK";
            this->buttonOK->UseVisualStyleBackColor = true;
            this->buttonOK->Click += gcnew System::EventHandler(this, &SettingsDialog::buttonOK_Click);
            //
            // checkBoxVSync
            //
            this->checkBoxVSync->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->checkBoxVSync->AutoSize = true;
            this->checkBoxVSync->Checked = true;
            this->checkBoxVSync->CheckState = System::Windows::Forms::CheckState::Checked;
            this->checkBoxVSync->Location = System::Drawing::Point(109, 375);
            this->checkBoxVSync->Name = L"checkBoxVSync";
            this->checkBoxVSync->Size = System::Drawing::Size(57, 17);
            this->checkBoxVSync->TabIndex = 3;
            this->checkBoxVSync->Text = L"VSync";
            this->checkBoxVSync->UseVisualStyleBackColor = true;
            //
            // buttonBgnd
            //
            this->buttonBgnd->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
            this->buttonBgnd->Location = System::Drawing::Point(109, 22);
            this->buttonBgnd->Name = L"buttonBgnd";
            this->buttonBgnd->Size = System::Drawing::Size(125, 23);
            this->buttonBgnd->TabIndex = 6;
            this->buttonBgnd->Text = L"Set Color";
            this->buttonBgnd->UseVisualStyleBackColor = true;
            this->buttonBgnd->Click += gcnew System::EventHandler(this, &SettingsDialog::buttonBgnd_Click);
            //
            // groupBox1
            //
            this->groupBox1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->groupBox1->Controls->Add(this->checkBoxGradient);
            this->groupBox1->Controls->Add(this->buttonBgnd);
            this->groupBox1->Location = System::Drawing::Point(12, 477);
            this->groupBox1->Name = L"groupBox1";
            this->groupBox1->Size = System::Drawing::Size(240, 61);
            this->groupBox1->TabIndex = 6;
            this->groupBox1->TabStop = false;
            this->groupBox1->Text = L"Background";
            //
            // checkBoxGradient
            //
            this->checkBoxGradient->AutoSize = true;
            this->checkBoxGradient->Location = System::Drawing::Point(8, 25);
            this->checkBoxGradient->Name = L"checkBoxGradient";
            this->checkBoxGradient->Size = System::Drawing::Size(66, 17);
            this->checkBoxGradient->TabIndex = 5;
            this->checkBoxGradient->Text = L"Gradient";
            this->checkBoxGradient->UseVisualStyleBackColor = true;
            //
            // groupBox2
            //
            this->groupBox2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
                | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->groupBox2->Controls->Add(this->checkBoxVSync);
            this->groupBox2->Controls->Add(this->checkBoxFS);
            this->groupBox2->Controls->Add(this->listBoxRes);
            this->groupBox2->Location = System::Drawing::Point(12, 10);
            this->groupBox2->Name = L"groupBox2";
            this->groupBox2->Size = System::Drawing::Size(240, 400);
            this->groupBox2->TabIndex = 7;
            this->groupBox2->TabStop = false;
            this->groupBox2->Text = L"Screen Settings";
            //
            // groupBox3
            //
            this->groupBox3->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->groupBox3->Controls->Add(this->comboBoxAPI);
            this->groupBox3->Location = System::Drawing::Point(12, 416);
            this->groupBox3->Name = L"groupBox3";
            this->groupBox3->Size = System::Drawing::Size(240, 55);
            this->groupBox3->TabIndex = 8;
            this->groupBox3->TabStop = false;
            this->groupBox3->Text = L"API";
            //
            // comboBoxAPI
            //
            this->comboBoxAPI->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->comboBoxAPI->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->comboBoxAPI->FormattingEnabled = true;
            this->comboBoxAPI->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"LWN", L"OpenGL" });
            this->comboBoxAPI->Location = System::Drawing::Point(8, 19);
            this->comboBoxAPI->Name = L"comboBoxAPI";
            this->comboBoxAPI->Size = System::Drawing::Size(226, 21);
            this->comboBoxAPI->TabIndex = 4;
            //
            // SettingsDialog
            //
            this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
            this->ClientSize = System::Drawing::Size(264, 579);
            this->Controls->Add(this->groupBox3);
            this->Controls->Add(this->groupBox2);
            this->Controls->Add(this->groupBox1);
            this->Controls->Add(this->buttonOK);
            this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
            this->MinimumSize = System::Drawing::Size(280, 500);
            this->Name = L"SettingsDialog";
            this->Text = L"Tear Test Settings";
            this->groupBox1->ResumeLayout(false);
            this->groupBox1->PerformLayout();
            this->groupBox2->ResumeLayout(false);
            this->groupBox2->PerformLayout();
            this->groupBox3->ResumeLayout(false);
            this->ResumeLayout(false);

        }
#pragma endregion
private: System::Void buttonOK_Click(System::Object^  sender, System::EventArgs^  e) {
    if (listBoxRes->SelectedIndex >= 0)
    {
        resString = settingsResString;
        char *str = (char*)(void*)Marshal::StringToHGlobalAnsi(listBoxRes->SelectedItem->ToString());
        strcpy(resString, str);
        Marshal::FreeHGlobal((System::IntPtr)str);
    }

    fullscreen = checkBoxFS->Checked;
    vsync = checkBoxVSync->Checked;
    tt.GetScene().bgnd.gradient = checkBoxGradient->Checked;
    apiString = (comboBoxAPI->SelectedIndex == 0) ? "LWN" : "OpenGL";

    Close();
}
private: System::Void buttonBgnd_Click(System::Object^  sender, System::EventArgs^  e) {
    ColorDialog ^c = gcnew ColorDialog;
    if (c->ShowDialog() == Windows::Forms::DialogResult::OK)
    {
        tt.GetScene().bgnd.color[0] = (1.0f / 255.0f) * c->Color.R;
        tt.GetScene().bgnd.color[1] = (1.0f / 255.0f) * c->Color.G;
        tt.GetScene().bgnd.color[2] = (1.0f / 255.0f) * c->Color.B;
    }
}
};
}
