<!-- XSLT script to colwert doxygen enum xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
  <html>
  <body>
    <!-- Load all doxgen generated xml files -->
    <xsl:for-each select="//memberdef[@kind='enum']">
        <b>Enum <xsl:value-of select="name"/></b>
        <xsl:for-each select="briefdescription/para">
            <p style="margin-left:80px"><xsl:value-of select="."/></p>
        </xsl:for-each>
        <xsl:for-each select="detaileddescription/para">
            <p style="margin-left:80px"><xsl:value-of select='./node()[not(self::jama-implements)]'/></p>
        </xsl:for-each>
        <xsl:if test="enumvalue">
            <p style="margin-left:80px">
                <i>Values:</i>
            </p>
        </xsl:if>
        <xsl:for-each select="enumvalue">
            <p style="margin-left:80px">
                <b><xsl:value-of select="name"/></b>
                <xsl:if test="initializer">
                    <xsl:text> </xsl:text><xsl:value-of select="initializer"/>
                </xsl:if>
                <p style="margin-left:120px">
                  <xsl:value-of select="briefdescription"/>
                  <xsl:value-of select="detaileddescription"/>
                </p>
            </p>
        </xsl:for-each>
        <br/>
        <h4 style="margin-left:40px;color:#76b900;">File Location</h4>
        <p style="margin-left:80px">
            <xsl:value-of select="location/@file"/>
        </p>
    </xsl:for-each>
  </body>
  </html>
  </xsl:template>
</xsl:stylesheet>
