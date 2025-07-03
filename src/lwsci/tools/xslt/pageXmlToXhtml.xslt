<!-- XSLT script to colwert doxygen blanket statements xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
    <html>
      <body>
        <xsl:for-each select="//compounddef[@kind='page']/detaileddescription/sect1">
          <xsl:sort select="title"/>

            <xsl:if test="not(title = preceding::title)">
              <br/>
              <b><xsl:value-of select="title"/></b>
            </xsl:if>

            <xsl:for-each select="para">
              <p style="margin-left:20px"><xsl:value-of select="text()"/></p>
              <xsl:for-each select="itemizedlist/listitem">
                <ul style="margin-left:40px">
                  <li>
                    <xsl:value-of select="para"/>
                  </li>
                </ul>
            </xsl:for-each>
          </xsl:for-each>

      </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
