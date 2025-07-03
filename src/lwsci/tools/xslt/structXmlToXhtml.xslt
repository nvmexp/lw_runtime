<!-- XSLT script to colwert doxygen struct xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
  <html>
  <body>
    <!-- Load all doxgen generated xml files -->
    <xsl:for-each select="//compounddef[@kind='struct']">
      <b>struct <xsl:value-of select="compoundname"/></b>
      <xsl:for-each select="briefdescription/para">
        <p style="margin-left:40px"><xsl:value-of select="."/></p>
      </xsl:for-each>
      <xsl:for-each select="detaileddescription/para">
        <p style="margin-left:40px"><xsl:value-of select='./node()[not(self::jama-implements)]'/></p>
      </xsl:for-each>
      <xsl:if test="sectiondef[@kind='public-func']/memberdef[@kind='function']">
        <h3 style="margin-left:40px">Public Functions</h3>
      </xsl:if>
      <p style="margin-left:40px">
        <xsl:for-each select="sectiondef[@kind='public-func']/memberdef[@kind='function']">
          <p style="margin-left:40px">
              <xsl:value-of select="type"/>
              <xsl:text> </xsl:text>
              <b><xsl:value-of select="name"/></b>
              <!-- TODO: Format argstring below by replacing it <param siblings> -->
              <xsl:value-of select="argsstring"/>
          </p>
          <p style="margin-left:80px">
            <xsl:if test="briefdescription/para">
              <p style="margin-left:80px"><xsl:value-of select="briefdescription/para"/></p>
            </xsl:if>
            <xsl:for-each select="detaileddescription/para">
              <xsl:if test="not(parameterlist)">
                <p style="margin-left:80px"><xsl:value-of select="."/></p>
              </xsl:if>
              <xsl:if test="parameterlist[@kind='param']">
                <p style="margin-left:80px"><b>Parameters</b></p>
              </xsl:if>
              <xsl:for-each select="parameterlist[@kind='param']/parameteritem">
                <ul style="margin-left:120px">
                  <li>
                    [<xsl:value-of select="parameternamelist/parametername/@direction"/>]
                    <xsl:text> </xsl:text>
                    <xsl:value-of select="parameternamelist/parametername"/>
                    <xsl:text>: </xsl:text>
                    <xsl:value-of select="parameterdescription/para"/>
                  </li>
                </ul>
              </xsl:for-each>
            </xsl:for-each>
          </p>
        </xsl:for-each>
      </p>
      <br/>
      <xsl:if test="sectiondef[@kind='public-attrib']/memberdef[@kind='variable']">
        <b style="margin-left:40px">Public Members</b>
      </xsl:if>
      <xsl:for-each select="sectiondef[@kind='public-attrib']/memberdef[@kind='variable']">
        <p style="margin-left:40px">
            <xsl:value-of select="type"/>
            <xsl:text> </xsl:text>
            <b><xsl:value-of select="name"/>
            <xsl:value-of select="argsstring"/>
            </b>
            <!-- TODO: Format argstring below by replacing it <param siblings> -->
            <p style="margin-left:80px">
              <xsl:value-of select="briefdescription/para"/>
              <xsl:value-of select="detaileddescription"/>
            </p>
        </p>
      </xsl:for-each>
      <h4 style="margin-left:40px;color:#76b900;">File Location</h4>
      <p style="margin-left:80px">
        <xsl:value-of select="location/@file"/>
      </p>
    </xsl:for-each>
  </body>
  </html>
  </xsl:template>
</xsl:stylesheet>
